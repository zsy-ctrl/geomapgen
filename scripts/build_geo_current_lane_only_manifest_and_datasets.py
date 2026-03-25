import sys
from typing import List

from build_geo_current_manifest_and_datasets import main as build_main


def _forwarded_argv(argv: List[str]) -> List[str]:
    out: List[str] = []
    for token in list(argv):
        if token == "--include-intersection-boundary":
            # lane-only 入口强制关闭路口边界监督。
            continue
        out.append(token)
    if "--include-lane" not in out:
        out.append("--include-lane")
    return out


def main() -> None:
    original = list(sys.argv[1:])
    forwarded = _forwarded_argv(original)
    sys.argv = [sys.argv[0]] + forwarded
    build_main()


if __name__ == "__main__":
    main()
