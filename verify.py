"""Verify the compact package using only the Python standard library."""

from pathlib import Path
import hashlib, json

root = Path(__file__).resolve().parent
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for name, expected in manifest["files_sha256"].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit("Missing or modified file: " + name)
print("Verified", len(manifest["files_sha256"]), "files.")
print("Four .pt files are saved model weights, not Python scripts.")
