import os
import zipfile
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models" / "distilbert-base-uncased"
ZIP_PATH = BASE_DIR / "model.zip"
MODEL_URL = os.environ.get("MODEL_URL")


def ensure_model() -> None:
    """
    Ensuring the model exists locally by downloading and extracting it
    from MODEL_URL if missing.
    """

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print("Model already exists so skipping download.")
        return

    if not MODEL_URL:
        raise RuntimeError(
            "MODEL_URL environment variable not set. "
            "Set it to a direct Dropbox download link."
        )
    
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("Downloading model from:", MODEL_URL)
    urllib.request.urlretrieve(MODEL_URL, ZIP_PATH)

    print("Extracting model...")
    try:
        with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(MODEL_DIR.parent)
    except zipfile.BadZipFile:
        raise RuntimeError("Downloaded model ZIP is corrupted or invalid.")

    ZIP_PATH.unlink(missing_ok=True)

    print(f"Model ready at: {MODEL_DIR}")


if __name__ == "__main__":
    ensure_model()
