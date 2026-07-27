"""Print the fully-resolved HEDGE-C config, following the _base_ inheritance
chain (hedge_c_base.yaml -> real_heavy.yaml -> default.yaml), as JSON.

Lets a reader inspect exactly what ran for the paper without tracing three
YAML files by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiments.runner import load_config  # noqa: E402

CONFIG_PATH = ROOT / "configs" / "hedge_c_base.yaml"


def main() -> None:
    config = load_config(CONFIG_PATH)
    print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
