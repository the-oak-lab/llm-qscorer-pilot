"""The HTTP API without the network: auth, validation, the batch limit, and a full predict with a fake engine."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from qscorer.api import MAX_QUESTIONS, create_app
from tests.test_baseline import raw

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "spacing-exp4" / "model.json"
TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def question(i):
    return {"id": f"q{i}", "question": {"stem": f"What is {i} + 1?", "choices": [{"label": c, "text": f"{i + k}"} for k, c in enumerate("abcd")]},
            "answerKey": "b", "feedback-text": f"{i} + 1 = {i + 1}.", "type": "ignored"}


class Engine:
    """Every feature present for even questions, absent for odd ones; q-fail always errors. Answers in the shape of the
    scorer's endpoint (chat completions for gemini-flash, decisions for jev) and records the model asked for."""
    def __init__(self, item_ids, scorer, model):
        self.item_ids, self.scorer, self.model = item_ids, scorer, model

    def describe(self):
        return {"backend": "fake"}

    def many(self, jobs, log, passes, accept, progress):
        done = {}
        for j in jobs:
            qid = j.id.split("|")[0]
            if qid == "q-fail":
                r = raw(j.id, error="HTTPStatusError: 500")
            else:
                present = int(qid[1:]) % 2 == 0
                if self.scorer == "jev":
                    r = {"id": j.id, "model": self.model + "-20261014", "usage": {"input_tokens": 500, "output_tokens": 40, "cost": 2e-5},
                         "answers": {i: {"type": "noul", "noul": 0.9 if present else 0.1} for i in json.loads(j.prompt)["questions"]}}
                else:
                    r = raw(j.id, text=json.dumps({"features": [{"id": i, "present": present} for i in self.item_ids]})) | {"model": self.model}
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            if accept(r):
                done[j.id] = r
        return done


@pytest.fixture(scope="module")
def client():
    item_ids = [it["id"] for it in json.loads(MODEL.read_text())["items"]]
    return TestClient(create_app(MODEL, TOKEN, engine_factory=lambda scorer, model: Engine(item_ids, scorer, model)))


def test_auth(client):
    assert client.get("/health").status_code == 200
    body = {"questions": [question(0)]}
    assert client.post("/predict", json=body).status_code == 401
    assert client.post("/predict", json=body, headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/features").status_code == 401
    assert client.get("/features", headers=AUTH).status_code == 200


def test_validation(client):
    def post(body):
        return client.post("/predict", json=body, headers=AUTH).status_code
    assert post({"questions": []}) == 422
    assert post({"questions": [question(i) for i in range(MAX_QUESTIONS + 1)]}) == 422
    assert post({"questions": [question(0), question(0)]}) == 422
    assert post({"questions": [question(0) | {"answerKey": "z"}]}) == 422
    assert post({"questions": [question(0)], "samples": 4}) == 422
    assert post({"questions": [question(0)], "unknown": 1}) == 422
    assert post({"questions": [question(0)], "scorer": "gpt"}) == 422


def test_predict(client):
    body = {"questions": [question(i) for i in range(MAX_QUESTIONS - 1)] + [{**question(0), "id": "q-fail"}], "samples": 2}
    r = client.post("/predict", json=body, headers=AUTH)
    assert r.status_code == 200
    out = r.json()
    assert [p["id"] for p in out["predictions"]] == [f"q{i}" for i in range(MAX_QUESTIONS - 1)]
    assert out["failed"] == [{"id": "q-fail", "error": "HTTPStatusError: 500"}]
    assert out["calls"] == 2 * MAX_QUESTIONS
    assert out["scorer"] == "gemini-flash" and out["scoring_model"] == "google/gemini-3.8-flash"
    assert out["served_by"] == {"google/gemini-3.8-flash": 2 * (MAX_QUESTIONS - 1)}
    even, odd = out["predictions"][0], out["predictions"][1]
    assert even["calls_answered"] == 2 and set(even["scores"].values()) == {1.0}
    assert set(odd["scores"].values()) == {0.0} and odd["raising"] == [] and odd["lowering"] == []
    assert even["class"] in ("higher", "lower") and 0 <= even["percentile"] <= 1
    assert even["raising"] and all(c["contribution"] > 0 for c in even["raising"])
    assert all(c["contribution"] < 0 for c in even["lowering"])
    model = json.loads(MODEL.read_text())
    assert odd["predicted_slope"] == round(model["intercept"], 4)


def test_all_failed(client):
    r = client.post("/predict", json={"questions": [{**question(0), "id": "q-fail"}]}, headers=AUTH)
    assert r.status_code == 502


def test_schemas_are_current():
    sys.path.insert(0, str(ROOT / "scripts"))
    from export_api_schemas import schemas
    for name, s in schemas().items():
        path = ROOT / "schemas" / f"{name}.schema.json"
        assert json.loads(path.read_text()) == s, f"{path} is stale: run python scripts/export_api_schemas.py"


def test_scorer_jev(client):
    r = client.post("/predict", json={"questions": [question(0)], "scorer": "jev"}, headers=AUTH)
    assert r.status_code == 200
    out = r.json()
    assert out["scorer"] == "jev" and out["scoring_model"] == "typesafe/jev-1.14" and out["served_by"] == {"typesafe/jev-1.14-20261014": 1}
    p = out["predictions"][0]
    # the tuned variant: every probability 0.9, then the Gemini calibration (thresholds at most 0.9 mark it present)
    assert p["calls_answered"] == 1 and out["cost"] == 2e-5 and all(0 <= v <= 1 for v in p["scores"].values())
    assert sum(v == 1.0 for v in p["scores"].values()) >= 40
