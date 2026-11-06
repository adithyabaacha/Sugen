"""Download the cross-encoder during Docker build."""
from pathlib import Path
from sentence_transformers import CrossEncoder

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L6-v2"
REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
DEST = Path("/app/models/cross_encoder")

if __name__ == "__main__":
    model = CrossEncoder(
        MODEL_ID,
        revision=REVISION,
        device="cpu"
    )

    DEST.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(
        str(DEST),
        create_model_card=False
    )

    assert (DEST / "config.json").exists()
    assert any(
        (DEST / f).exists()
        for f in ("model.safetensors", "pytorch_model.bin")
    )

    print("Cross-encoder saved for offline inference at", DEST)
