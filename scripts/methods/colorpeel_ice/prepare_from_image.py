"""Automatically discover objects, segment instances and initialize four concepts each."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.colorpeel_ice.image_intake import main


if __name__ == "__main__":
    sys.exit(main())
