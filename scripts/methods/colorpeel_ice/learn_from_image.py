"""Single-image Scheme A preparation and selected independent branch training."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.colorpeel_ice.single_image_pipeline import main

if __name__ == '__main__':
    sys.exit(main())
