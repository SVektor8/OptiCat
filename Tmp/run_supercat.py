from pathlib import Path
import os
import subprocess
import sys

# Allow running the script directly from repository root without editable install.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _tk_preflight_ok() -> bool:
    if sys.platform != "darwin":
        return True
    if os.environ.get("OPTICAT_SKIP_TK_PREFLIGHT") == "1":
        return True

    env = dict(os.environ)
    env["OPTICAT_SKIP_TK_PREFLIGHT"] = "1"
    cmd = [sys.executable, "-c", "import tkinter as tk; r=tk.Tk(); r.destroy()"]
    probe = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return probe.returncode == 0


if __name__ == "__main__":
    if sys.platform == "darwin" and sys.version_info >= (3, 13):
        print(
            "Python 3.13 on macOS is currently unstable with Tkinter/Tk in this setup.\n"
            "Please run OptiCat with Python 3.12."
        )
        sys.exit(1)

    if not _tk_preflight_ok():
        print(
            "Tkinter/Tk crashed during startup in this Python environment.\n"
            "This is a runtime issue (often Python 3.13 + Tk 8.6 on macOS), not OptiCat code.\n"
            "Use Python 3.12 (recommended) and recreate the venv, then run again."
        )
        sys.exit(1)

    from opticat.gui import SuperCat

    SuperCat().open()
