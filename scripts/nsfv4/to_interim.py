"""Convert the nsfv4 DataShop export into the interim transaction file the slope stage reads.
Step Name := mcq-<CF (Question Id)>; Problem View := dense rank of Problem Start Time per (student, problem);
the schedule Condition Name (2nd column of that name) kept as `Condition Name`, the algorithm one renamed; survey rows dropped.
Usage: python to_interim.py <export.tsv> <out.tsv>"""
import sys
import pandas as pd
src, dst = sys.argv[1], sys.argv[2]
tx = pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False)
tx = tx.rename(columns={"Condition Name": "Condition Name (Algorithm)", "Condition Type": "Condition Type (Algorithm)",
                        "Condition Name.1": "Condition Name", "Condition Type.1": "Condition Type"})
n0 = len(tx)
tx = tx[tx["CF (Question Type)"] == "Multiple Choice"].copy()
tx["Step Name"] = "mcq-" + tx["CF (Question Id)"]
start = pd.to_datetime(tx["Problem Start Time"])
tx["Problem View"] = start.groupby([tx["Anon Student Id"], tx["Problem Name"]]).rank(method="dense").astype(int).astype(str)
tx.to_csv(dst, sep="\t", index=False)
print(f"{n0} rows -> {len(tx)} MCQ rows; {tx['Step Name'].nunique()} questions; {tx['Anon Student Id'].nunique()} students; "
      f"rows per (student, problem view, step): max {tx.groupby(['Anon Student Id', 'Problem Name', 'Problem View', 'Step Name']).size().max()}")
