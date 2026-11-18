"""Kaggle kernel hw3-01-annotations: annotation table (no text) for the full dataset."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hw3-ai-text-detection_Kolodchak.git"
SRC = Path("/tmp/repo")
CODE = SRC / "code"
LOG = Path("/kaggle/working/annotations.log")


def run(*args):
    res = subprocess.run([sys.executable, *args], cwd=CODE, capture_output=True, text=True)
    text = f"\n$ python {' '.join(args)}\n{res.stdout}{res.stderr[-3000:]}"
    print(text)
    with open(LOG, "a") as f:
        f.write(text)
    if res.returncode != 0:
        raise SystemExit(f"failed: {args}")


def main():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
    run("scripts/build_annotations.py")  # writes /kaggle/working/annotations/annotations.csv.gz


if __name__ == "__main__":
    main()
