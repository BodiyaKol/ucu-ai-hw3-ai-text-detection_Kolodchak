import numpy as np

from src.config import load_config, set_seed
from src.data.io import LABELS


def test_load_config_merge(tmp_path):
    (tmp_path / "base.yaml").write_text("seed: 1\nmodel:\n  lr: 0.1\n  epochs: 2\n")
    (tmp_path / "child.yaml").write_text("base: base.yaml\nmodel:\n  lr: 0.01\n")
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg == {"seed": 1, "model": {"lr": 0.01, "epochs": 2}}


def test_set_seed_reproducible():
    set_seed(42)
    a = np.random.rand(3)
    set_seed(42)
    b = np.random.rand(3)
    assert np.allclose(a, b)


def test_protocol_loads():
    cfg = load_config("configs/protocol.yaml")
    assert cfg["seed"] == 42
    assert cfg["labels"] == LABELS
