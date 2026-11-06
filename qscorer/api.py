"""HTTP API: score a batch of questions against the checklist and return the fitted model's predictions.

Run:  QSCORER_API_TOKEN=... uvicorn qscorer.api:app --host 0.0.0.0 --port 8000
Env:  QSCORER_API_TOKEN (required; clients send `Authorization: Bearer <token>`), QSCORER_MODEL (default
      models/spacing-exp4/model.json), OPENROUTER_API_KEY (or a .env file, as for the scripts).

`POST /predict` takes up to MAX_QUESTIONS questions, sends their scoring calls in parallel (one round for a full
batch with one sample) and answers when every call has finished. The handler is a plain `def`, so FastAPI runs it
in a worker thread and `run_presence` can start its own event loop there. Each request logs its raw responses in a
temporary folder that is deleted afterwards. The request and response shapes are the pydantic models below;
`scripts/export_api_schemas.py` writes them as JSON Schema to `schemas/`.
"""
from __future__ import annotations

import json
import os
import secrets
import tempfile
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, model_validator

from qscorer.config import PresenceParams
from qscorer.llm import ROOT, Backend, make_backend
from qscorer.predict import apply
from qscorer.presence import run_presence

MAX_QUESTIONS = 20
MAX_SAMPLES = 3
DEFAULT_MODEL = ROOT / "models" / "spacing-exp4" / "model.json"


# ---------------------------------------------------------------- request

class Choice(BaseModel):
    model_config = ConfigDict(extra="allow")
    label: str = Field(min_length=1, description="the option's label, e.g. a")
    text: str = Field(min_length=1, description="the option's text")


class Stem(BaseModel):
    model_config = ConfigDict(extra="allow")
    stem: str = Field(min_length=1, description="the question stem")
    choices: list[Choice] = Field(min_length=2, description="the answer options")


class Question(BaseModel):
    """One question in the KCluster format plus `feedback-text`. Other fields are allowed and ignored."""
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    id: str = Field(min_length=1, description="a unique id for the question")
    question: Stem
    answerKey: str = Field(min_length=1, description="the label of the correct option")
    feedback_text: str = Field("", alias="feedback-text", description="the feedback students see after answering")

    @model_validator(mode="after")
    def key_is_an_option(self) -> Question:
        labels = [c.label for c in self.question.choices]
        if len(set(labels)) != len(labels):
            raise ValueError(f"question {self.id}: option labels are not unique")
        if self.answerKey not in labels:
            raise ValueError(f"question {self.id}: answerKey {self.answerKey!r} is not one of the option labels {labels}")
        return self


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    questions: list[Question] = Field(min_length=1, max_length=MAX_QUESTIONS, description=f"1 to {MAX_QUESTIONS} questions")
    samples: int = Field(1, ge=1, le=MAX_SAMPLES, description="scoring calls per question; a feature's score is the share of calls that found it present")
    context: str | None = Field(None, min_length=1, description="the sentence that tells the model how the questions are used (default: the training dataset's)")

    @model_validator(mode="after")
    def ids_are_unique(self) -> PredictRequest:
        ids = [q.id for q in self.questions]
        if dupes := sorted({i for i in ids if ids.count(i) > 1}):
            raise ValueError(f"question ids are not unique: {dupes}")
        return self


# ---------------------------------------------------------------- response

class Contribution(BaseModel):
    feature: str = Field(description="the feature id, e.g. f07")
    name: str
    contribution: float = Field(description="score times coefficient, in the units of predicted_slope")


class Prediction(BaseModel):
    id: str
    predicted_slope: float = Field(description="predicted gain in accuracy with practice, in logits per log-attempt (training median about 0.24)")
    percentile: float = Field(ge=0, le=1, description="share of the training questions with a lower prediction")
    class_: Literal["higher", "lower"] = Field(alias="class", description="higher or lower than the training median")
    calls_answered: int = Field(ge=1, description="scoring calls that returned a usable answer")
    raising: list[Contribution] = Field(description="features present that raised the prediction, largest first")
    lowering: list[Contribution] = Field(description="features present that lowered the prediction, largest first")
    scores: dict[str, float] = Field(description="the score on each checklist feature, 0 to 1")


class Failure(BaseModel):
    id: str
    error: str = Field(description="the last error of the question's scoring calls")


class PredictResponse(BaseModel):
    model: str = Field(description="the dataset the predictor was trained on")
    scoring_model: str = Field(description="the LLM that scored the questions")
    predictions: list[Prediction] = Field(description="one per question that was scored, in request order")
    failed: list[Failure] = Field(description="questions with no usable scoring call")
    calls: int
    tokens: dict[str, int]


class Feature(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str
    name: str
    direction: str
    coefficient: float


# ---------------------------------------------------------------- app

def contributions(model: dict, scores: dict[str, float], sign: int) -> list[Contribution]:
    names = {it["id"]: it["name"] for it in model["items"]}
    c = [(i, s * model["coefficients"][i]) for i, s in scores.items() if s > 0]
    c = [(i, v) for i, v in c if np.sign(v) == sign]
    return [Contribution(feature=i, name=names[i], contribution=round(v, 4)) for i, v in sorted(c, key=lambda t: -abs(t[1]))]


def last_errors(log: Path) -> dict[str, str]:
    """The last error logged for each question. Failed calls stay out of `run_presence`'s records, so the log is the only place they are."""
    out: dict[str, str] = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip() and (rec := json.loads(line)).get("error"):
                out[rec["id"].split("|")[0]] = str(rec["error"])
    return out


def create_app(model_path: Path | None = None, token: str | None = None,
               engine_factory: Callable[[PresenceParams], Backend] | None = None) -> FastAPI:
    token = token or os.environ.get("QSCORER_API_TOKEN")
    if not token:
        raise RuntimeError("set QSCORER_API_TOKEN")
    model_path = Path(model_path or os.environ.get("QSCORER_MODEL") or DEFAULT_MODEL)
    model = json.loads(model_path.read_text())
    scoring = model["scoring"]
    item_ids = [it["id"] for it in model["items"]]
    model_id = scoring["model"] if "/" in scoring["model"] else "google/" + scoring["model"]
    engine_factory = engine_factory or make_backend

    bearer = HTTPBearer(auto_error=False)

    def authorize(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if creds is None or not secrets.compare_digest(creds.credentials.encode(), token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    app = FastAPI(title="llm-qscorer", version="0.1.0",
                  description="Predict how much students will improve on multiple-choice questions with practice.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model["dataset"]}

    @app.get("/features", dependencies=[Depends(authorize)], response_model=list[Feature])
    def features() -> list[dict]:
        return model["features"]

    @app.post("/predict", dependencies=[Depends(authorize)], response_model=PredictResponse, response_model_by_alias=True)
    def predict(req: PredictRequest) -> PredictResponse:
        questions = {q.id: q.model_dump(by_alias=True) for q in req.questions}
        params = PresenceParams(backend="openrouter", model=model_id, thinking=scoring["thinking"], temperature=scoring["temperature"],
                                samples=req.samples, concurrency=MAX_QUESTIONS * req.samples, max_output_tokens=8192,
                                timeout=120.0, passes=2)
        with tempfile.TemporaryDirectory(prefix="qscorer-") as tmp:
            result = run_presence(questions, list(questions), model["items"], req.context or scoring["context"], scoring["system"],
                                  params, Path(tmp), engine=engine_factory(params), progress=lambda s: None)
            errors = last_errors(Path(tmp) / "responses.jsonl")
        scored = result.scores
        predictions: list[Prediction] = []
        if len(scored):
            table = apply(model, scored).set_index("id")
            n_ok = scored.set_index("id")["n_ok"]
            for qid in questions:
                if qid not in table.index:
                    continue
                row = table.loc[qid]
                s = {i: float(row[i]) for i in item_ids}
                predictions.append(Prediction.model_validate({
                    "id": qid, "predicted_slope": round(float(row["predicted_slope"]), 4), "percentile": float(row["percentile"]),
                    "class": row["class"], "calls_answered": int(n_ok[qid]),
                    "raising": contributions(model, s, 1), "lowering": contributions(model, s, -1), "scores": s}))
        done = {p.id for p in predictions}
        failed = [Failure(id=qid, error=errors.get(qid, "no usable answer")) for qid in questions if qid not in done]
        if not predictions:
            raise HTTPException(status.HTTP_502_BAD_GATEWAY, {"message": "no question could be scored", "failed": [f.model_dump() for f in failed]})
        return PredictResponse(model=model["dataset"], scoring_model=model_id, predictions=predictions, failed=failed,
                               calls=result.summary["calls"], tokens=result.summary["tokens"])

    return app


def __getattr__(name: str):
    """`qscorer.api:app` for uvicorn, built on first access so importing the module needs no token."""
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)
