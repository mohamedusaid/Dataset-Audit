import subprocess
import sys
from pathlib import Path

DATASET_URL = input("Enter dataset ZIP URL: ").strip()
ZIP_PATH = Path("/content/dataset.zip")
OUTPUT = Path("/content/audit_results")

if not DATASET_URL:
    raise ValueError("Dataset URL cannot be empty.")

subprocess.run(
    ["wget", "-c", "-O", str(ZIP_PATH), DATASET_URL],
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        "requirements.txt",
    ],
    check=True,
)

subprocess.run(
    [
        sys.executable,
        "-m",
        "audit_cli",
        "--data",
        str(ZIP_PATH),
        "--output",
        str(OUTPUT),
    ],
    check=True,
)

print("\nAudit complete.")
print(f"Results: {OUTPUT}")
