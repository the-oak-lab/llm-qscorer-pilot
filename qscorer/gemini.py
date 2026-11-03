"""A thin asynchronous Gemini client for the Gemini stages.

Every call is one job (an id, the contents, a config dict for `GenerateContentConfig`). Jobs run
concurrently under a semaphore, with a small start-delay ramp; a call that raises comes back as
an error record instead of sinking the batch, and a call that hangs fails after `timeout`
seconds. The raw response of every call (thought parts, text parts, finish reason, token
counts) is appended to a JSONL log as it arrives, so an interrupted run resumes from the log,
and failed or unparsable calls are resent in later passes. `parse` reduces a raw record to the
answer text, the thought summary and the token counts.
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

ROOT = Path(__file__).resolve().parents[1]


def api_key(env_file: Path = ROOT / ".env") -> str:
    """`GEMINI_API_KEY` from the environment, else from the `.env` file."""
    if key := os.environ.get("GEMINI_API_KEY"):
        return key
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("no GEMINI_API_KEY in the environment or .env")


@dataclass
class Job:
    id: str
    contents: Any           # a string, or google-genai contents
    config: dict            # fields of GenerateContentConfig


@dataclass
class Response:
    id: str
    text: str | None
    thought: str | None
    finish_reason: str | None
    prompt_tokens: int | None
    output_tokens: int | None
    thought_tokens: int | None
    error: str | None

    @property
    def ok(self) -> bool:
        return self.error is None and self.text is not None


def parse(raw: dict) -> Response:
    """The answer text (non-thought parts joined), the thought summary (thought parts joined),
    the finish reason and the token counts of one raw record."""
    if raw.get("error"):
        return Response(raw["id"], None, None, None, None, None, None, raw["error"])
    candidates = raw.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
    thought = "".join(p.get("text", "") for p in parts if p.get("thought")) or None
    usage = raw.get("usage_metadata") or {}
    finish = candidates[0].get("finish_reason") if candidates else None
    return Response(raw["id"], text, thought, finish, usage.get("prompt_token_count"), usage.get("candidates_token_count"),
                    usage.get("thoughts_token_count"), None if text is not None else f"no text (finish reason {finish})")


class Gemini:
    def __init__(self, model: str, key: str | None = None, concurrency: int = 8, delay: float = 0.05, timeout: float = 180.0):
        self.client = genai.Client(api_key=key or api_key())
        self.model, self.delay, self.timeout = model, delay, timeout
        self.semaphore = asyncio.Semaphore(concurrency)
        self._next_delay = 0.0

    async def one(self, job: Job) -> dict:
        """One call; the raw response as a dict carrying the job id, or `{"id", "error"}`."""
        self._next_delay += self.delay
        await asyncio.sleep(self._next_delay)
        config = types.GenerateContentConfig.model_validate(job.config | {"http_options": {"timeout": int(self.timeout * 1000)}})
        async with self.semaphore:
            try:
                response = await self.client.aio.models.generate_content(model=self.model, contents=job.contents, config=config)
                d = response.to_json_dict()
                d.pop("sdk_http_response", None)  # headers and status, not part of the answer
                return {"id": job.id} | d
            except Exception as e:  # one failed call must not sink the batch
                return {"id": job.id, "error": f"{type(e).__name__}: {e}"}

    async def many(self, jobs: list[Job], log: Path, passes: int = 3, accept: Callable[[dict], bool] | None = None,
                   progress: Callable[[str], None] = print) -> dict[str, dict]:
        """Run the jobs, appending every raw record to `log` as it arrives, and resend the ones
        that failed `accept` (default: no error) in up to `passes` rounds. Records already in the
        log that pass `accept` are not asked again, so an interrupted run resumes."""
        accept = accept or (lambda raw: not raw.get("error"))
        done: dict[str, dict] = {}
        if log.exists():
            for line in log.read_text().splitlines():
                if line.strip():
                    raw = json.loads(line)
                    if accept(raw):
                        done[raw["id"]] = raw
        wanted = {j.id for j in jobs}
        done = {k: v for k, v in done.items() if k in wanted}
        for attempt in range(passes):
            todo = [j for j in jobs if j.id not in done]
            if not todo:
                break
            progress(f"pass {attempt + 1}: {len(todo)} calls ({len(done)} already done)")
            self._next_delay = 0.0
            if attempt:
                await asyncio.sleep(2.0 ** attempt)
            with open(log, "a") as f:
                for coro in asyncio.as_completed([self.one(j) for j in todo]):
                    raw = await coro
                    f.write(json.dumps(raw) + "\n")
                    f.flush()
                    if accept(raw):
                        done[raw["id"]] = raw
        return done


def run(jobs: list[Job], log: Path, model: str, concurrency: int, delay: float, timeout: float, passes: int,
        accept: Callable[[dict], bool] | None = None, progress: Callable[[str], None] = print) -> dict[str, dict]:
    """`Gemini.many` from synchronous code."""
    engine = Gemini(model, concurrency=concurrency, delay=delay, timeout=timeout)
    return asyncio.run(engine.many(jobs, log, passes, accept, progress))
