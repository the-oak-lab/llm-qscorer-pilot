"""The language-model layer shared by every stage that calls a model.

A stage builds `Job`s (an id, the prompt, the system text, the JSON schema of the answer and the
sampling settings) and hands them to a `Backend`, which turns each job into one call in its own
wire format and gives back the raw response as a dict. Two backends exist: `qscorer.gemini`
(the Gemini API through google-genai) and `qscorer.openrouter` (the OpenAI chat-completions
format, as served by OpenRouter and by any server of that format). `make_backend` picks one from
a stage's parameters.

The shared part lives here: jobs run concurrently under a semaphore with a small start-delay
ramp; a call that raises comes back as an error record instead of sinking the batch, and a call
that hangs fails after `timeout` seconds. The raw response of every call (answer, thought summary,
finish reason, token counts, seconds taken, which backend) is appended to a JSONL log as it
arrives, so an interrupted run resumes from the log, and failed or unparsable calls are resent in
later passes. `parse` reduces a raw record of either backend to the answer text, the thought
summary and the token counts, so the stages never see a wire format.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
THINKING = ("minimal", "low", "medium", "high")


def api_key(name: str, env_file: Path | None = None) -> str:
    """The key named `name` (e.g. `GEMINI_API_KEY`) from the environment, else from a `.env` file:
    the given one, the working directory's, or the repository's."""
    if key := os.environ.get(name):
        return key
    for path in ([env_file] if env_file else []) + [Path.cwd() / ".env", ROOT / ".env"]:
        if path.exists():
            for line in path.read_text().splitlines():
                if line.startswith(f"{name}="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError(f"no {name} in the environment or in a .env file")


@dataclass
class Job:
    """One call. The answer is JSON text following `schema` when a schema is given, free text
    otherwise. `thinking` is the reasoning level (`minimal`, `low`, `medium`, `high`), or None for
    the model's default; `thoughts` asks for the thought summary to be returned and logged."""
    id: str
    prompt: str
    system: str = ""
    schema: dict | None = None
    temperature: float = 1.0
    seed: int | None = None
    max_output_tokens: int = 4096
    thinking: str | None = "medium"
    thoughts: bool = True

    def settings(self) -> dict:
        """The sampling settings, for example files and summaries."""
        return {"temperature": self.temperature, "seed": self.seed, "max_output_tokens": self.max_output_tokens,
                "thinking": self.thinking, "thoughts": self.thoughts}


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
    """The answer text, the thought summary, the finish reason and the token counts of one raw
    record, whichever backend wrote it. `output_tokens` counts the answer without the thoughts in
    both formats (the OpenAI format's `completion_tokens` includes reasoning, so the reasoning
    count is taken off when it is reported)."""
    if raw.get("error"):
        return Response(raw["id"], None, None, None, None, None, None, str(raw["error"]))
    if "choices" in raw:
        return _parse_openai(raw)
    return _parse_gemini(raw)


def _parse_gemini(raw: dict) -> Response:
    candidates = raw.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    text = "".join(p.get("text", "") for p in parts if not p.get("thought")) or None
    thought = "".join(p.get("text", "") for p in parts if p.get("thought")) or None
    usage = raw.get("usage_metadata") or {}
    finish = candidates[0].get("finish_reason") if candidates else None
    return Response(raw["id"], text, thought, finish, usage.get("prompt_token_count"), usage.get("candidates_token_count"),
                    usage.get("thoughts_token_count"), None if text is not None else f"no text (finish reason {finish})")


def _parse_openai(raw: dict) -> Response:
    choices = raw.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    text = message.get("content") or None
    thought = message.get("reasoning") or message.get("reasoning_content") or None
    if not thought and message.get("reasoning_details"):
        thought = "".join(d.get("text") or d.get("summary") or "" for d in message["reasoning_details"] if isinstance(d, dict)) or None
    if text and text.startswith("<thought>") and "</thought>" in text:  # Gemini's OpenAI-compatible endpoint inlines the summary
        inline, text = text[len("<thought>"):].split("</thought>", 1)
        thought, text = thought or inline.strip() or None, text.strip() or None
    finish = choices[0].get("finish_reason") if choices else None
    usage = raw.get("usage") or {}
    prompt, completion, total = usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens")
    reasoning = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if reasoning is None and None not in (prompt, completion, total) and total > prompt + completion:
        reasoning = total - prompt - completion  # reported only as the gap (Gemini's OpenAI-compatible endpoint)
        output = completion
    else:
        output = completion - reasoning if completion is not None and reasoning is not None else completion
    return Response(raw["id"], text, thought, finish, prompt, output, reasoning,
                    None if text is not None else f"no text (finish reason {finish})")


class Backend:
    """What the stages use. A subclass makes one call (`one`) and one embedding request
    (`embed_batch`) in its wire format; the concurrency, the logging, the resumption and the
    retries are here."""
    name = "backend"
    library = ""

    def __init__(self, model: str, concurrency: int = 8, delay: float = 0.05, timeout: float = 180.0):
        self.model, self.concurrency, self.delay, self.timeout = model, concurrency, delay, timeout
        self._next_delay = 0.0

    async def one(self, job: Job) -> dict:
        """One call; the raw response as a dict carrying the job id, or `{"id", "error"}`."""
        raise NotImplementedError

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        """One embedding request: one vector per text, in order."""
        raise NotImplementedError

    def describe(self) -> dict:
        """What to record about the backend in a run's summary."""
        return {"backend": self.name, "model": self.model, "library": self.library,
                "version": metadata.version(self.library) if self.library else None}

    async def _timed(self, job: Job) -> dict:
        self._next_delay += self.delay
        await asyncio.sleep(self._next_delay)
        async with self.semaphore:
            start = time.monotonic()
            try:
                d = await self.one(job)
            except Exception as e:  # one failed call must not sink the batch
                d = {"error": f"{type(e).__name__}: {e}"}
            if "id" in d:  # the OpenAI format names its completion `id`; the record's id is the job's
                d["response_id"] = d.pop("id")
            return {"id": job.id, "backend": self.name, "elapsed": round(time.monotonic() - start, 2)} | d

    async def many_async(self, jobs: list[Job], log: Path, passes: int = 3, accept: Callable[[dict], bool] | None = None,
                         progress: Callable[[str], None] = print) -> dict[str, dict]:
        """Run the jobs, appending every raw record to `log` as it arrives, and resend the ones
        that failed `accept` (default: no error) in up to `passes` rounds. Records already in the
        log that pass `accept` are not asked again, so an interrupted run resumes."""
        accept = accept or (lambda raw: not raw.get("error"))
        self.semaphore = asyncio.Semaphore(self.concurrency)
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
                for coro in asyncio.as_completed([self._timed(j) for j in todo]):
                    raw = await coro
                    f.write(json.dumps(raw) + "\n")
                    f.flush()
                    if accept(raw):
                        done[raw["id"]] = raw
        return done

    def many(self, jobs: list[Job], log: Path, passes: int = 3, accept: Callable[[dict], bool] | None = None,
             progress: Callable[[str], None] = print) -> dict[str, dict]:
        """`many_async` from synchronous code."""
        return asyncio.run(self.many_async(jobs, log, passes, accept, progress))

    async def embed_async(self, texts: list[str], model: str, batch: int = 100, concurrency: int = 8, prefix: str = "",
                          progress: Callable[[str], None] = print) -> list[list[float]]:
        """One embedding per text, in requests of `batch` texts, `concurrency` requests in flight;
        `prefix` (e.g. a task instruction) goes in front of every text. A failed request is
        retried up to three times."""
        semaphore = asyncio.Semaphore(concurrency)
        chunks = [texts[i:i + batch] for i in range(0, len(texts), batch)]

        async def one(k: int, chunk: list[str]) -> tuple[int, list[list[float]]]:
            async with semaphore:
                for attempt in range(3):
                    try:
                        return k, await self.embed_batch(model, [prefix + t for t in chunk])
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2.0 ** attempt)
                raise RuntimeError("unreachable")

        out: dict[int, list[list[float]]] = {}
        for coro in asyncio.as_completed([one(k, c) for k, c in enumerate(chunks)]):
            k, vectors = await coro
            out[k] = vectors
            if len(out) % 10 == 0 or len(out) == len(chunks):
                progress(f"embedded {sum(len(v) for v in out.values())} of {len(texts)}")
        return [v for k in range(len(chunks)) for v in out[k]]

    def embed(self, texts: list[str], model: str, batch: int = 100, concurrency: int = 8, prefix: str = "",
              progress: Callable[[str], None] = print) -> list[list[float]]:
        """`embed_async` from synchronous code."""
        return asyncio.run(self.embed_async(texts, model, batch, concurrency, prefix, progress))


def make_backend(params: Any) -> Backend:
    """The backend a stage's parameters ask for: `params.backend` is `gemini` or `openrouter`,
    with `params.model`, `concurrency`, `delay`, `timeout` and, for `openrouter`, `base_url`."""
    kind = getattr(params, "backend", "gemini")
    if kind == "gemini":
        from qscorer.gemini import Gemini
        return Gemini(params.model, concurrency=params.concurrency, delay=params.delay, timeout=params.timeout)
    if kind == "openrouter":
        from qscorer.openrouter import OpenRouter
        return OpenRouter(params.model, base_url=getattr(params, "base_url", None), concurrency=params.concurrency, delay=params.delay,
                          timeout=params.timeout)
    raise ValueError(f"unknown backend {kind!r}")
