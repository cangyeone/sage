from pathlib import Path

import pytest


MODEL_DIR = Path("open_models/bge-m3")


def test_bge_m3_flag_model_loads():
    """Smoke-test local BGE-M3 loading when the optional model is present."""
    if not (MODEL_DIR / "config.json").exists():
        pytest.skip("Optional local model open_models/bge-m3 is not present")

    from FlagEmbedding import BGEM3FlagModel

    BGEM3FlagModel(
        str(MODEL_DIR),
        use_fp16=True,
        device="cpu",
    )
