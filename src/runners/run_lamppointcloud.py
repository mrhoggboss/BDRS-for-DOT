"""LAMPPointCloud: LAMP's matrix-free path on Point Cloud instances. The
orchestrator runs ``run_<name in lower case>``, so this module is the entry
point; the runner is ``run_lamp``."""

import sys

from src.runners.run_lamp import ALGO_PC, main_for


def main(argv=None) -> int:
    return main_for(ALGO_PC, argv)


if __name__ == "__main__":
    sys.exit(main())
