"""Kaggle kernel hw3-06-xlmr-v2: XLM-R, 100k pool, 3 epochs, random chunks (T4)."""
import subprocess
import sys
from pathlib import Path

REPO = "https://github.com/BodiyaKol/ucu-ai-hw3-ai-text-detection_Kolodchak.git"
SRC = Path("/tmp/repo")
CODE = SRC / "code"
LOG = Path("/kaggle/working/xlmr_v2.log")


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
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=False)
    run("scripts/train_encoder.py", "--config", "configs/encoder/xlmr_v2.yaml")


if __name__ == "__main__":
    main()
