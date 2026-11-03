"""The OpenRouter backend: the OpenAI chat-completions format over HTTP, key `OPENROUTER_API_KEY`.

A job becomes one `POST /chat/completions` with a system and a user message, `response_format`
of type `json_schema` when the job has a schema, the temperature, the seed, the output budget and
OpenRouter's `reasoning` object (effort = the thinking level; `exclude` when thought summaries are
not wanted). The raw record is the response body (choices with the content and the reasoning,
finish reason, usage with the reasoning token count). Embeddings are `POST /embeddings` with the
texts as a list. The model id is OpenRouter's (e.g. `google/gemini-3.8-flash`).

`base_url` points the same code at any other server of this format (give it up to the version
path, e.g. `https://host/v1`), with the key `OPENAI_API_KEY`. The `reasoning` object is
OpenRouter's own, so other servers get the standard `reasoning_effort` field instead and return
the summary as `reasoning_content` (LiteLLM gateways do) or not at all. One server is
special-cased because it is the test target that needs no OpenRouter key: Gemini's own
OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai`, key
`GEMINI_API_KEY`). It rejects `seed` and `reasoning`, takes the thinking level and the thought
summaries through `extra_body.google.thinking_config`, returns the summary inline in the content
between `<thought>` tags, and reports no reasoning token count (the count is the gap between the
total and the prompt plus the completion). `qscorer.llm.parse` knows all these shapes.
"""
from __future__ import annotations

import asyncio

import httpx

from qscorer.llm import Backend, Job, api_key

OPENROUTER_URL = "https://openrouter.ai/api/v1"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


class OpenRouter(Backend):
    name = "openrouter"
    library = "httpx"

    def __init__(self, model: str, base_url: str | None = None, key: str | None = None, key_env: str | None = None,
                 concurrency: int = 8, delay: float = 0.05, timeout: float = 180.0):
        super().__init__(model, concurrency, delay, timeout)
        self.base_url = (base_url or OPENROUTER_URL).rstrip("/")
        self.dialect = ("gemini" if "generativelanguage.googleapis.com" in self.base_url
                        else "openrouter" if self.base_url == OPENROUTER_URL else "openai")
        self.key = key or api_key(key_env or {"gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}[self.dialect])
        self._client: httpx.AsyncClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def describe(self) -> dict:
        return super().describe() | {"base_url": self.base_url, "dialect": self.dialect}

    def client(self) -> httpx.AsyncClient:
        """One client per event loop: every `many` or `embed` runs its own loop, and a pool of
        connections cannot outlive the loop it was opened on."""
        loop = asyncio.get_running_loop()
        if self._client is None or self._loop is not loop:
            self._client = httpx.AsyncClient(base_url=self.base_url, headers={"Authorization": f"Bearer {self.key}"}, timeout=self.timeout)
            self._loop = loop
        return self._client

    def body(self, job: Job) -> dict:
        """The request body of a job."""
        messages = ([{"role": "system", "content": job.system}] if job.system else []) + [{"role": "user", "content": job.prompt}]
        b: dict = {"model": self.model, "messages": messages, "temperature": job.temperature, "max_completion_tokens": job.max_output_tokens}
        if job.schema is not None:
            b["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": job.schema}}
        if self.dialect == "gemini":
            thinking: dict = {"include_thoughts": job.thoughts}
            if job.thinking:
                thinking["thinking_level"] = job.thinking
            b["extra_body"] = {"google": {"thinking_config": thinking}}
            return b
        if job.seed is not None:
            b["seed"] = job.seed
        if self.dialect == "openrouter":
            reasoning: dict = {"exclude": not job.thoughts}
            if job.thinking:
                reasoning["effort"] = job.thinking
            b["reasoning"] = reasoning
        elif job.thinking:
            b["reasoning_effort"] = job.thinking
        return b

    async def one(self, job: Job) -> dict:
        r = await self.client().post("/chat/completions", json=self.body(job))
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text[:500]}"}
        d = r.json()
        if not isinstance(d, dict) or d.get("error"):  # some provider failures come back with status 200
            return {"error": str(d.get("error") if isinstance(d, dict) else d)[:500]}
        return d

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        r = await self.client().post("/embeddings", json={"model": model, "input": texts})
        r.raise_for_status()
        data = r.json()["data"]
        if len(data) != len(texts):
            raise RuntimeError(f"{len(data)} embeddings for {len(texts)} texts")
        return [e["embedding"] for e in sorted(data, key=lambda e: e.get("index") or 0)]
