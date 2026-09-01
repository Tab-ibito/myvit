"""在终端中查看 checkpoint 摘要。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myvit.checkpoint import inspect_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="查看 myvit checkpoint 摘要")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    summary = inspect_checkpoint(args.checkpoint)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

