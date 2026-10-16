"""Download the public datasets used by the experiments into data/raw/.

Breast Cancer Wisconsin ships with scikit-learn. MAGIC Gamma Telescope
(OpenML 1120) and Spambase (OpenML 44) are fetched from OpenML and cached as
CSV. All three are UCI datasets released under CC BY 4.0.

Usage: python scripts/download_data.py [--data-dir data/raw] [--force]
"""

from __future__ import annotations

import argparse

from cshap.data import DATASETS, download_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()
    for name in args.datasets or list(DATASETS):
        info = DATASETS[name]
        if info.openml_id is None:
            print(f"{name}: bundled with scikit-learn, nothing to download")
            continue
        path = download_dataset(name, args.data_dir, force=args.force)
        print(f"{name}: OpenML {info.openml_id} -> {path}")


if __name__ == "__main__":
    main()
