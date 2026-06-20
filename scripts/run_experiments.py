"""Run the full experiment matrix end-to-end.

Steps:
  1. Preprocess both corpora under `drop_exact_duplicates` mode (within-dataset
     deduplication; cross-dataset overlap is left intact as a property of the
     natural synthetic pipeline — see documentation/data-pipeline.md).
  2. Run the dataset audit (provenance, overlap, class balance, etc.).
  3. Run the full training matrix:
     30 seeds × 3 classifiers × 3 strategies = 270 runs.
  4. Generate evaluation figures.
  5. Persist the best model + metadata under outputs/models/.

Outputs land in outputs/{metrics,figures,tables,models}/.

Usage:
    .venv/bin/python -m scripts.run_experiments
"""

import time

from src.data.audit import run_audit
from src.data.preprocessing import run_preprocessing
from src.models.figures import generate_all_figures
from src.models.train import run_experiments

MODE = "drop_exact_duplicates"


def main() -> None:
    t0 = time.time()

    print(f"[1/4] Preprocessing (mode={MODE})...", flush=True)
    manual, synthetic, _combined = run_preprocessing(mode=MODE, save=True)
    print(f"  manual rows: {len(manual)} ({manual['label'].value_counts().to_dict()})")
    print(f"  synthetic rows: {len(synthetic)} ({synthetic['label'].value_counts().to_dict()})")

    print("[2/4] Auditing datasets...", flush=True)
    run_audit(manual, synthetic, save=True)

    print("[3/4] Running training matrix (3 strategies × 3 models × 30 seeds = 270 runs)...", flush=True)
    metrics_df = run_experiments(
        manual, synthetic, duplicate_mode=MODE, verbose=True,
    )

    print("[4/4] Generating figures...", flush=True)
    generate_all_figures(metrics_df, manual, synthetic)

    print(f"\nDone in {(time.time() - t0)/60:.1f} min. Outputs in outputs/")


if __name__ == "__main__":
    main()
