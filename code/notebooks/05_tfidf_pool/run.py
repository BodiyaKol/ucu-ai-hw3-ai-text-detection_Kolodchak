"""Kaggle kernel hw3-05-tfidf-pool: TF-IDF on 150k pool texts (CPU)."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hw3-ai-text-detection_Kolodchak.git"
SRC = Path("/tmp/repo")
CODE = SRC / "code"
LOG = Path("/kaggle/working/tfidf_pool.log")


def run(*args):
    """Stream output live to the kernel log and to encoder.log."""
    with open(LOG, "a") as f:
        f.write(f"\n$ python {' '.join(args)}\n")
        proc = subprocess.Popen([sys.executable, "-u", *args], cwd=CODE,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            f.write(line)
            f.flush()
        if proc.wait() != 0:
            raise SystemExit(f"failed: {args}")


def main():
    subprocess.run(["git", "clone", "--depth", "1", REPO, str(SRC)], check=True)
    run("scripts/train_tfidf.py", "--config", "configs/tfidf/pool.yaml")


if __name__ == "__main__":
    main()
