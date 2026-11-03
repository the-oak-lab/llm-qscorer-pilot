"""The model layer without the network: both raw-record shapes parse to the same `Response`, each
backend builds the request its API expects from one `Job`, and the Gemini request is the one the
logged runs were made with."""
import json

from qscorer.gemini import Gemini
from qscorer.llm import Backend, Job, parse
from qscorer.openrouter import GEMINI_URL, OpenRouter

JOB = Job("q1|improve|2", "the prompt", system="the system", schema={"type": "object"}, temperature=1.0, seed=2, max_output_tokens=4096,
          thinking="medium", thoughts=True)


def test_parse_gemini_record():
    raw = {"id": "a", "backend": "gemini", "elapsed": 1.0,
           "candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}, {"text": '{"answer": 1}'}]}, "finish_reason": "STOP"}],
           "usage_metadata": {"prompt_token_count": 100, "candidates_token_count": 20, "thoughts_token_count": 50}}
    r = parse(raw)
    assert r.ok and r.text == '{"answer": 1}' and r.thought == "hmm" and r.finish_reason == "STOP"
    assert (r.prompt_tokens, r.output_tokens, r.thought_tokens) == (100, 20, 50)


def test_parse_openrouter_record():
    raw = {"id": "a", "backend": "openrouter", "elapsed": 1.0,
           "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": '{"answer": 1}', "reasoning": "hmm"}}],
           "usage": {"prompt_tokens": 100, "completion_tokens": 70, "total_tokens": 170, "completion_tokens_details": {"reasoning_tokens": 50}}}
    r = parse(raw)
    assert r.ok and r.text == '{"answer": 1}' and r.thought == "hmm" and r.finish_reason == "stop"
    assert (r.prompt_tokens, r.output_tokens, r.thought_tokens) == (100, 20, 50)  # the answer's tokens, without the reasoning
    details = {"choices": [{"finish_reason": "stop", "message": {"content": "x", "reasoning_details": [{"type": "reasoning.text", "text": "a"}, {"type": "reasoning.summary", "summary": "b"}]}}]}
    assert parse({"id": "b"} | details).thought == "ab"


def test_parse_gemini_compatible_record():
    """Gemini's OpenAI-compatible endpoint: the summary inline between <thought> tags, the reasoning count as the usage gap."""
    raw = {"id": "a", "choices": [{"finish_reason": "stop", "message": {"content": "<thought>**Plan**\n\nhmm\n\n</thought>" + '{"answer": 1}'}}],
           "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 170}}
    r = parse(raw)
    assert r.ok and r.text == '{"answer": 1}' and r.thought == "**Plan**\n\nhmm" and (r.prompt_tokens, r.output_tokens, r.thought_tokens) == (100, 20, 50)
    plain = {"id": "a", "choices": [{"finish_reason": "stop", "message": {"content": '{"answer": 1}'}}], "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}}
    r = parse(plain)
    assert r.ok and r.thought is None and r.thought_tokens is None and r.output_tokens == 20


def test_parse_failures():
    assert parse({"id": "a", "error": "HTTP 429: slow down"}).error == "HTTP 429: slow down"
    r = parse({"id": "a", "choices": [{"finish_reason": "length", "message": {"content": None}}], "usage": {}})
    assert not r.ok and r.error == "no text (finish reason length)"
    r = parse({"id": "a", "candidates": [{"content": {"parts": [{"text": "t", "thought": True}]}, "finish_reason": "MAX_TOKENS"}]})
    assert not r.ok and r.thought == "t" and "MAX_TOKENS" in r.error


def test_gemini_config_is_the_logged_one():
    """The request the Stage 1 and 2 runs were logged with, field for field."""
    assert Gemini.config(JOB) == {"system_instruction": "the system", "response_mime_type": "application/json", "response_json_schema": {"type": "object"},
                                  "temperature": 1.0, "seed": 2, "max_output_tokens": 4096, "thinking_config": {"thinking_level": "MEDIUM", "include_thoughts": True}}
    plain = Job("x", "p", thinking=None, thoughts=False)
    assert Gemini.config(plain) == {"temperature": 1.0, "max_output_tokens": 4096}


def test_openrouter_body():
    b = OpenRouter("google/gemini-3.8-flash", key="k").body(JOB)
    assert b["model"] == "google/gemini-3.8-flash" and b["messages"] == [{"role": "system", "content": "the system"}, {"role": "user", "content": "the prompt"}]
    assert b["response_format"] == {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {"type": "object"}}}
    assert b["seed"] == 2 and b["max_completion_tokens"] == 4096 and b["temperature"] == 1.0 and b["reasoning"] == {"exclude": False, "effort": "medium"}
    assert "extra_body" not in b
    quiet = OpenRouter("m", key="k").body(Job("x", "p", thinking=None, thoughts=False))
    assert quiet["reasoning"] == {"exclude": True} and "seed" not in quiet and "response_format" not in quiet and quiet["messages"][0]["role"] == "user"
    json.dumps(b)


def test_gemini_compatible_body():
    e = OpenRouter("gemini-3.8-flash", base_url=GEMINI_URL + "/", key="k")
    assert e.dialect == "gemini" and e.base_url == GEMINI_URL and e.describe()["base_url"] == GEMINI_URL
    b = e.body(JOB)
    assert "seed" not in b and "reasoning" not in b and b["extra_body"] == {"google": {"thinking_config": {"include_thoughts": True, "thinking_level": "medium"}}}


def test_other_openai_server_body():
    e = OpenRouter("gemini/gemini-3.8-flash", base_url="https://gateway.example.edu/v1", key="k")
    assert e.dialect == "openai"
    b = e.body(JOB)
    assert b["seed"] == 2 and b["reasoning_effort"] == "medium" and "reasoning" not in b and "extra_body" not in b
    assert "reasoning_effort" not in e.body(Job("x", "p", thinking=None))


class Echo(Backend):
    """Answers every job at once; its responses carry the OpenAI format's own `id`."""
    name = "echo"

    async def one(self, job):
        return {"id": "chatcmpl-123", "choices": [{"finish_reason": "stop", "message": {"content": job.prompt}}], "usage": {}}


def test_many_keeps_the_job_id_and_resumes(tmp_path):
    log = tmp_path / "responses.jsonl"
    jobs = [Job("a", "1"), Job("b", "2")]
    done = Echo("m", concurrency=2, delay=0).many(jobs, log, passes=2, progress=lambda s: None)
    assert set(done) == {"a", "b"} and done["a"]["response_id"] == "chatcmpl-123" and done["a"]["backend"] == "echo"
    assert parse(done["b"]).text == "2"
    lines = log.read_text().splitlines()
    assert len(lines) == 2 and {json.loads(l)["id"] for l in lines} == {"a", "b"}
    again = Echo("m").many(jobs + [Job("c", "3")], log, passes=1, progress=lambda s: None)
    assert set(again) == {"a", "b", "c"} and len(log.read_text().splitlines()) == 3  # a and b were not asked again
