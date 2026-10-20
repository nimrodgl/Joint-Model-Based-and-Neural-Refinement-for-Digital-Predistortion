"""Generate V05 Tables I--V and Figures 1--2 from the retained result records.

This exports archived numerical evidence; it does not retrain or replay every
benchmark model. Use run.py for the representative saved-model replay.
"""

import argparse
from datetime import datetime
from pathlib import Path

from jmbnr.data import ROOT
from jmbnr.reporting import generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or ROOT / "outputs" / (
        "v05-results-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=False)
    generate(output)
    print("Generated Tables I--V and Figures 1--2.", flush=True)
    print("Outputs location:", output.resolve(), flush=True)


if __name__ == "__main__":
    main()
