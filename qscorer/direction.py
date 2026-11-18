"""Stage 2.3 follow-up (`direction`): which way Gemini claimed each checklist item works.

Discovery asked for beneficial and for detrimental features in separate calls (`CONDITIONS` in
`qscorer/batches.py`), and Stage 2.3 grouped the raw features into checklist items. The grouping
puts a feature together with its opposite: "explicit distractor refutation in feedback" and
"absence of distractor rationales in feedback" both belong to the refutational-feedback item. So
the polarity a call asked for tells what Gemini claimed about the raw feature, not about the item.

One call per item and batch of its raw features says, for each raw feature, whether it describes
the item ("same"), the item's absence or reverse ("opposite"), or something else ("neither").
Gemini claimed that the item's presence helps when it proposed a "same" feature as beneficial or
an "opposite" one as detrimental, and that it harms in the other two cases. Per item, the result
gives the counts, `claimed_good_share`: the share of its "same" and "opposite" raw features
whose claim is that the item's presence helps, and `bad_batches`: the number of discovery
batches in which at least one claim says it harms. One batch can give several raw features
(several conditions and samples), so batches count independent evidence better than features.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from qscorer.batches import CONDITIONS
from qscorer.canon import feature_text
from qscorer.config import CanonParams
from qscorer.llm import Job, make_backend, parse

RELATIONS = ("same", "opposite", "neither")
PROMPT = (
    "Below is one item of a checklist for scoring multiple-choice science questions and their feedback, followed by {n} features that an "
    "earlier analysis described and grouped under this item.\n\n"
    "Checklist item: {name}. {definition} {criterion}\n\n"
    "Features:\n{items}\n\n"
    "For each feature, say how it relates to the checklist item:\n"
    '- "same": a question with this feature has the checklist item;\n'
    '- "opposite": the feature is the absence or the reverse of the checklist item, so a question with this feature lacks the item;\n'
    '- "neither": the feature describes something else, or covers both the item and its absence.\n\n'
    'Respond in JSON with one field, "features": a list with one object per feature, each with "index" (the feature\'s number) and '
    '"relation" ("same", "opposite" or "neither").')
SCHEMA = {"type": "object", "required": ["features"], "properties": {"features": {"type": "array", "items": {
    "type": "object", "required": ["index", "relation"],
    "properties": {"index": {"type": "integer"}, "relation": {"type": "string", "enum": list(RELATIONS)}}}}}}


def build_jobs(features: pd.DataFrame, checklist: list[dict], params: CanonParams, system: str, batch: int = 60) -> tuple[list[Job], dict[str, list[int]]]:
    """The calls, and for each call the row positions in `features` it covers, in order."""
    texts = feature_text(features).to_numpy()
    jobs, rows = [], {}
    for it in checklist:
        members = [k for k, item in enumerate(features["item"]) if item == it["id"]]
        for start in range(0, len(members), batch):
            chunk = members[start:start + batch]
            items = "\n".join(f"{i + 1}. {texts[k]}" for i, k in enumerate(chunk))
            prompt = PROMPT.format(n=len(chunk), name=it["name"], definition=it["definition"].strip(), criterion=it["criterion"].strip(), items=items)
            jid = f"{it['id']}|{start}"
            jobs.append(Job(jid, prompt, system=system, schema=SCHEMA, temperature=params.temperature, seed=0,
                            max_output_tokens=params.max_output_tokens, thinking=params.thinking))
            rows[jid] = chunk
    return jobs, rows


def parse_relations(text: str | None, n: int) -> dict[int, str] | None:
    """index (1-based) -> relation, when every index 1..n is answered; else None."""
    if not text:
        return None
    try:
        items = json.loads(text).get("features")
    except (ValueError, AttributeError):
        return None
    if not isinstance(items, list):
        return None
    out = {}
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("index"), int) and it.get("relation") in RELATIONS:
            out.setdefault(it["index"], it["relation"])
    return out if all(i in out for i in range(1, n + 1)) else None


def item_table(rows: pd.DataFrame, checklist: list[dict]) -> pd.DataFrame:
    """Per item: raw features, how many were judged same, opposite and neither, the claimed direction of the item's presence, and the
    number of discovery batches with at least one claim that it harms (the evidence behind a flaw)."""
    out = []
    for it in checklist:
        g = rows[rows["item"] == it["id"]]
        judged = g[g.relation.isin(["same", "opposite"])]
        out.append({"item": it["id"], "name": it["name"], "raw_features": len(g), "unjudged": int(g.relation.isna().sum()),
                    **{r: int((g.relation == r).sum()) for r in RELATIONS},
                    "claimed_good_share": float((judged.claim == "good").mean()) if len(judged) else float("nan"),
                    "bad_batches": int(judged[judged.claim == "bad"].batch.nunique())})
    return pd.DataFrame(out)


def run_direction(features: pd.DataFrame, checklist: list[dict], params: CanonParams, system: str, out: Path, batch: int = 60,
                  engine=None, progress=print) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Judge every raw feature against its item; raw log in `out/direction-responses.jsonl`, resumed if present.
    Returns the raw features with `asked`, `relation` and `claim` (what Gemini claimed about the item's presence: good, bad, or empty),
    the per-item table, and a summary."""
    jobs, rows = build_jobs(features, checklist, params, system, batch)
    def accept(raw: dict) -> bool:
        r = parse(raw)
        return r.ok and parse_relations(r.text, len(rows[r.id])) is not None
    out.mkdir(parents=True, exist_ok=True)
    engine = engine or make_backend(params)
    records = engine.many(jobs, out / "direction-responses.jsonl", params.passes, accept, progress)
    relation = pd.Series(None, index=range(len(features)), dtype=object)
    tokens = {"prompt_tokens": 0, "output_tokens": 0, "thought_tokens": 0}
    for jid, chunk in rows.items():
        r = parse(records[jid]) if jid in records else None
        answers = parse_relations(r.text, len(chunk)) if r else None
        if answers:
            for i, k in enumerate(chunk):
                relation[k] = answers[i + 1]
            for key in tokens:
                tokens[key] += getattr(r, key) or 0
    t = features[["batch", "condition", "sample", "feature", "name", "item"]].reset_index(drop=True).copy()
    t["asked"] = t.condition.map(lambda c: CONDITIONS[c][1])
    t["relation"] = relation.to_numpy()
    t["claim"] = [("good" if asked == "good" else "bad") if rel == "same" else ("bad" if asked == "good" else "good") if rel == "opposite" else None
                  for asked, rel in zip(t.asked, t.relation)]
    items = item_table(t, checklist)
    summary = {"raw_features": len(t), "calls": len(jobs), "answered": sum(1 for j in rows if j in records),
               "relations": t.relation.value_counts(dropna=False).to_dict(), "tokens": tokens, "batch": batch,
               "params": params.model_dump(mode="json"), "backend": engine.describe()}
    summary["relations"] = {str(k): int(v) for k, v in summary["relations"].items()}
    return t, items, summary
