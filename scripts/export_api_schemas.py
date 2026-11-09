"""Write the API's request and response shapes as JSON Schema files.

Run:  python scripts/export_api_schemas.py [--out schemas]
Out:  <out>/predict-request.schema.json, <out>/predict-response.schema.json, <out>/features-response.schema.json
"""
import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from qscorer.api import Feature, PredictRequest, PredictResponse

SCHEMAS = {
    "predict-request": (PredictRequest, "validation"),
    "predict-response": (PredictResponse, "serialization"),
    "features-response": (list[Feature], "serialization"),
}


def schemas() -> dict[str, dict]:
    out = {}
    for name, (shape, mode) in SCHEMAS.items():
        s = TypeAdapter(shape).json_schema(mode=mode, by_alias=True)
        out[name] = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"{name}.schema.json"} | s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="schemas")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, s in schemas().items():
        (out / f"{name}.schema.json").write_text(json.dumps(s, indent=2) + "\n")
        print(f"wrote {out}/{name}.schema.json")


if __name__ == "__main__":
    main()
