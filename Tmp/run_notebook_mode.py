from pathlib import Path
import sys

# Allow running the script directly from repository root without editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from opticat.core import SuperMan


if __name__ == "__main__":
    app = SuperMan(use_notebook_backend=True)
    app.ui()
