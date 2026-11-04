"""The Gemini backend: the Gemini API through google-genai, key `GEMINI_API_KEY`.

A job becomes one `generate_content` call: the system text, the answer's JSON schema, the
temperature, the seed, the output budget and the thinking level and thought summaries go in
`GenerateContentConfig`. The raw record is the response as the SDK serialises it (candidates with
thought and text parts, finish reason, usage metadata). Embeddings wrap each text as its own
`Content`, which is what makes `gemini-embedding-2` return one vector per text instead of one
vector for the lot.
"""
from __future__ import annotations

import asyncio

from google import genai
from google.genai import types

from qscorer.llm import Backend, Job, api_key


class Gemini(Backend):
    name = "gemini"
    library = "google-genai"

    def __init__(self, model: str, key: str | None = None, concurrency: int = 8, delay: float = 0.05, timeout: float = 180.0):
        super().__init__(model, concurrency, delay, timeout)
        self.key = key or api_key("GEMINI_API_KEY")
        self._client: genai.Client | None = None
        self._loop = None

    @property
    def client(self) -> genai.Client:
        """One client per event loop: every `many` or `embed` runs its own loop, and the async
        client's connection pool belongs to the loop it was first used in (a connection reused
        in a later loop fails with "Event loop is closed")."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if self._client is None or self._loop is not loop:
            self._client = genai.Client(api_key=self.key)
            self._loop = loop
        return self._client

    @staticmethod
    def config(job: Job) -> dict:
        """The `GenerateContentConfig` fields of a job."""
        c: dict = {"temperature": job.temperature, "max_output_tokens": job.max_output_tokens}
        if job.system:
            c["system_instruction"] = job.system
        if job.schema is not None:
            c["response_mime_type"] = "application/json"
            c["response_json_schema"] = job.schema
        if job.seed is not None:
            c["seed"] = job.seed
        thinking: dict = {}
        if job.thinking:
            thinking["thinking_level"] = job.thinking.upper()
        if job.thoughts:
            thinking["include_thoughts"] = True
        if thinking:
            c["thinking_config"] = thinking
        return c

    async def one(self, job: Job) -> dict:
        config = types.GenerateContentConfig.model_validate(self.config(job) | {"http_options": {"timeout": int(self.timeout * 1000)}})
        response = await self.client.aio.models.generate_content(model=self.model, contents=job.prompt, config=config)
        d = response.to_json_dict()
        d.pop("sdk_http_response", None)  # headers and status, not part of the answer
        return d

    async def embed_batch(self, model: str, texts: list[str]) -> list[list[float]]:
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in texts]
        r = await self.client.aio.models.embed_content(model=model, contents=contents)
        return [e.values for e in r.embeddings]
