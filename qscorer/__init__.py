"""LLM-based discovery of question features predictive of learning.

Stage 0.1 (`slopes`) estimates every question's practice slope from student data; the later
stages build on it. Every function takes data and parameters and returns plain data. The data
loading and the slope model were borrowed from the idea-22 project (`~/research/incubator/
idea-22-llm-cta`, package `llmcta`) and trimmed to what the design in `docs/design.md` uses."""
