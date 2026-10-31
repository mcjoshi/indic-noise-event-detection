"""Repository paths; override with VAANI_WORK, VAANI_DATA and VAANI_PRETRAINED."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = Path(os.environ.get("VAANI_WORK", ROOT / "work"))
DATA = Path(os.environ.get("VAANI_DATA", ROOT / "data"))
PRETRAINED = Path(os.environ.get("VAANI_PRETRAINED", ROOT / "pretrained"))
