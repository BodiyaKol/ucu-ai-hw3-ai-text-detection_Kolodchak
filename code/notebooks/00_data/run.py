"""Kaggle kernel hw3-00-data: full-data sanity check + mini sample for local CPU tests.

Clones the public repo, runs repo scripts on the competition data,
copies small outputs (mini CSVs, JSON reports, log) to /kaggle/working.
"""
import shutil
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hw3-ai-text-detection_Kolodchak.git"
SRC = Path("/tmp/repo")
CODE = SRC / "code"
WORK = Path("/kaggle/working")
LOG = WORK / "data_check.log"


def run(*args):
    """Run a repo script from code/, print + append its output to the log."""
    res = subprocess.run([sys.executable, *args], cwd=CODE, capture_output=True, text=True)
    text = f"\n$ python {' '.join(args)}\n{res.stdout}{res.stderr}"
    print(text)
    with open(LOG, "a") as f:
        f.write(text)
    if res.returncode != 0:
        raise SystemExit(f"failed: {args}")


def main():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
    run("scripts/check_data.py")
    run("scripts/make_mini.py")
    run("scripts/check_data.py", "--mini")

    shutil.copytree(CODE / "data" / "mini", WORK / "mini", dirs_exist_ok=True)
    for p in (CODE / "results").glob("m0_data_check*.json"):
        shutil.copy(p, WORK / p.name)
    print("outputs:", sorted(str(p.relative_to(WORK)) for p in WORK.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
