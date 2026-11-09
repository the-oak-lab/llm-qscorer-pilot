"""Extract the unique multiple-choice questions of a DataShop transaction TSV into the project's JSONL format.

The export has no option list, so the options are the distinct answers students submitted. The key is
CF (Exemplar Answer). Questions whose stem, options or feedback embed an image are left out. Options are
shuffled with a seed per question (the original order is not in the export) and labelled a, b, c, d.
Usage: python extract_questions.py <tsv> <all.jsonl> <sample.jsonl> <n> [seed]
"""
import csv, json, random, re, sys
from collections import Counter, defaultdict

csv.field_size_limit(10**9)
tsv, all_out, sample_out, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
r = csv.reader(open(tsv, newline=""), delimiter="\t")
h = next(r)
col = {name: h.index(name) for name in ["Input", "Feedback Text", "CF (Question Type)", "CF (Exemplar Answer)", "CF (Question Id)", "CF (Full Problem Name)", "CF (Timed Out)"]}
Q = defaultdict(lambda: {"inputs": Counter(), "key": Counter(), "fb": Counter(), "stem": Counter()})
for row in r:
    if row[col["CF (Question Type)"]] != "Multiple Choice":
        continue
    q = Q[row[col["CF (Question Id)"]]]
    if row[col["Input"]]:
        q["inputs"][row[col["Input"]]] += 1
    q["key"][row[col["CF (Exemplar Answer)"]]] += 1
    q["fb"][row[col["Feedback Text"]]] += 1
    q["stem"][row[col["CF (Full Problem Name)"]]] += 1

IMAGE = re.compile(r"!\[|<img", re.I)
out, skipped = [], Counter()
for qid, q in Q.items():
    stem, key, fb = (q[k].most_common(1)[0][0] for k in ("stem", "key", "fb"))
    opts = list(q["inputs"])
    if len(q["stem"]) > 1 or len(q["key"]) > 1 or len(q["fb"]) > 1:
        skipped["inconsistent"] += 1; continue
    if len(opts) != 4 or key not in opts:
        skipped["options incomplete"] += 1; continue
    if any(IMAGE.search(t) for t in [stem, fb, *opts]):
        skipped["image"] += 1; continue
    opts.sort(); random.Random(f"{seed}-{qid}").shuffle(opts)
    labels = "abcd"
    out.append({"id": f"mcq-{qid}", "question": {"stem": stem.strip(), "choices": [{"label": l, "text": t} for l, t in zip(labels, opts)]},
                "answerKey": labels[opts.index(key)], "feedback-text": fb.strip(), "type": "mcq"})
out.sort(key=lambda q: q["id"])
with open(all_out, "w") as f:
    f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in out)
sample = sorted(random.Random(seed).sample(out, min(n, len(out))), key=lambda q: q["id"])
with open(sample_out, "w") as f:
    f.writelines(json.dumps(q, ensure_ascii=False) + "\n" for q in sample)
print(f"{len(Q)} questions; {len(out)} kept; skipped {dict(skipped)}; {len(sample)} in the sample")
print("key positions in sample:", dict(Counter(q["answerKey"] for q in sample)))
