"""What a dataset is, and how a Stage 0.1 run is parameterised.

A dataset config (`configs/datasets/<name>.toml`) describes files in DataShop's formats: a
transaction file (or a student-step file) and a question JSONL. Run parameters have defaults
here and are overridden on the command line.
"""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]


class PhaseFilter(BaseModel):
    """Which rows enter the fit. Rows whose `column` matches `practice` (a regular expression)
    are practice opportunities, numbered per student and question in time order. Rows matching
    `baseline` (optional, e.g. a pretest) give each question a before-practice level; the change
    from that level to the first practice attempt is the question's jump. Rows matching neither
    (e.g. posttests) are left out."""
    column: str = "Level (Phase)"
    practice: str
    baseline: str | None = None


class TimedOutMarker(BaseModel):
    """How the log marks an attempt that timed out with no answer: `column` equals `value`
    and `input_column` is blank."""
    column: str = "CF (Timed Out)"
    value: str = "TRUE"
    input_column: str = "Input"


class PromptConfig(BaseModel):
    """What Gemini is told about how the questions were used. No numbers from the data."""
    system: str = "You are an expert in science education and the learning sciences."
    context: str = ("Students answered multiple-choice questions at a pretest and again during practice sessions spread over "
                    "several weeks. After each answer, students saw feedback and then moved on to other questions.")


class DatasetConfig(BaseModel):
    name: str
    transactions: Path
    questions: Path
    phases: PhaseFilter
    prompt: PromptConfig = PromptConfig()
    exclude: list[str] = Field(default_factory=list, description="question ids kept out of the Gemini stages (e.g. stems that reference a missing image)")
    timed_out: TimedOutMarker | None = None
    student_column: str = "Anon Student Id"
    step_column: str = "Step Name"
    set_column: str | None = Field(None, description="a transaction column (e.g. Class) whose set of values per question defines its item set")
    schedule_column: str | None = Field(None, description="a transaction column (e.g. Condition Name) giving the practice schedule of each (student, question) pair; needed for the schedule-adjusted fit")

    def resolved(self, root: Path) -> "DatasetConfig":
        """The same config with relative paths made absolute against `root`."""
        return self.model_copy(update={
            "transactions": root / self.transactions, "questions": root / self.questions})


class SlopeParams(BaseModel):
    """Parameters of Stage 0.1. A slope is in logits per log-attempt: the gain from practice
    attempt 1 to attempt k is slope * log k."""
    max_opportunities: int | None = Field(7, ge=1, description="keep each student's first k practice opportunities on a question (the baseline row is always kept); null keeps all")
    min_support: int = Field(10, ge=0, description="students with a second practice opportunity, below which a question gets no posterior")
    timeouts: Literal["keep", "drop"] = Field("keep", description="keep a no-answer timeout as an incorrect first attempt, or drop it")
    adjust_schedule: bool = Field(False, description="give each practice schedule its own first-practice level and mean slope (the sensitivity fit); the question's slope is then its deviation plus the row-weighted mean of the schedule slopes")
    reliability_splits: int = Field(5, ge=0, description="split-half repetitions; 0 skips the reliability check")
    seed: int = 0


class PairParams(BaseModel):
    """Parameters of Stage 0.2 (classification and matching) and 0.4 (its diagnostics)."""
    quantile: float = Field(1 / 3, gt=0, lt=0.5, description="within each item set, questions whose posterior mean slope is at or above the (1 - quantile) quantile are high-learning candidates, at or below the quantile low-learning")
    caliper: float = Field(0.10, gt=0, description="largest difference in observed first-practice accuracy (share correct, 0 to 1) within a pair")
    separation: float = Field(0.95, gt=0, lt=1, description="posterior probability that the high question's slope exceeds the low question's, required of a pair")
    min_difference: float = Field(0.20, ge=0, description="smallest difference of posterior mean slopes (logits per log-attempt) required of a pair")
    reliability_splits: int = Field(5, ge=0, description="split-half repetitions of the whole procedure (random halves of the students, refitted); 0 skips")
    seed: int = 0


class BaselineParams(BaseModel):
    """Parameters of Stage 1: Gemini's direct, no-checklist predictions, one question per call."""
    model: str = Field("gemini-3.8-flash", description="Gemini model id")
    prompts: list[Literal["improve", "known"]] = Field(["improve", "known"], description="the questions asked: the design's probability of improvement, and how many of 100 students already know the answer")
    samples: int = Field(3, ge=1, description="independent calls per question and prompt (each with its own seed)")
    temperature: float = Field(1.0, ge=0, description="sampling temperature; 1.0 is the model's default and the vendor's recommendation for Gemini 3 thinking models")
    thinking: Literal["minimal", "low", "medium", "high"] = Field("medium", description="Gemini thinking level")
    thoughts: bool = Field(True, description="ask for thought summaries and keep them per call")
    max_output_tokens: int = Field(4096, ge=64)
    concurrency: int = Field(8, ge=1, description="calls in flight at once")
    delay: float = Field(0.05, ge=0, description="start-delay ramp between calls, in seconds")
    timeout: float = Field(180.0, gt=0, description="seconds before a hanging call fails and is resent in the next pass")
    passes: int = Field(3, ge=1, description="rounds of resending failed or unparsable calls")
    limit: int | None = Field(None, ge=1, description="ask only this many questions, spread over the item sets (pilots)")
    seed: int = 0


def load_dataset_config(name_or_path: str | Path, root: Path = ROOT) -> DatasetConfig:
    """Load `configs/datasets/<name>.toml`, or a config file given by path, with paths resolved."""
    path = Path(name_or_path)
    if not path.suffix:
        path = root / "configs" / "datasets" / f"{name_or_path}.toml"
    with open(path, "rb") as f:
        return DatasetConfig(**tomllib.load(f)).resolved(root)
