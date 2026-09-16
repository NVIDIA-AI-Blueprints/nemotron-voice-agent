"""Generated protobuf modules live here."""

from __future__ import annotations

import sys
from pathlib import Path

generated_dir = Path(__file__).resolve().parent
generated_path = str(generated_dir)
if generated_path not in sys.path:
    sys.path.insert(0, generated_path)
