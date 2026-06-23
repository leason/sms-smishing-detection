"""Run the experiment matrix under the `overlap_aware` duplicate mode.

This is the robustness-check counterpart to `run_experiments.py`. The
overlap_aware mode does the normal within-dataset dedup, then additionally
removes synthetic rows whose raw text appears anywhere in the manual corpus.
Because the test set is a 20% holdout of the manual corpus, this eliminates
the exact-text overlap between synthetic training rows and manual test rows
that the natural-volume design retains.

Outputs land under `outputs/runs/overlap_aware/{metrics,tables,figures,models}/`
so the canonical headline outputs under `outputs/{...}/` are untouched. This is
achieved by monkey-patching the directory constants on `src.utils.paths` BEFORE
importing any consumer module (the `from src.utils.paths import X` bindings in
audit/train/evaluate/figures resolve to the patched values).

Usage:
    .venv/bin/python -m scripts.run_experiments_overlap_aware
"""

import time

from src.utils import paths as _paths

OVERLAP_RUN_DIR = _paths.PROJECT_ROOT / "outputs" / "runs" / "overlap_aware"
_paths.FIGURES_DIR = OVERLAP_RUN_DIR / "figures"
_paths.METRICS_DIR = OVERLAP_RUN_DIR / "metrics"
_paths.MODELS_DIR = OVERLAP_RUN_DIR / "models"
_paths.TABLES_DIR = OVERLAP_RUN_DIR / "tables"

from src.data.audit import run_audit  # noqa: E402
from src.data.preprocessing import run_preprocessing  # noqa: E402
from src.models.figures import generate_all_figures  # noqa: E402
from src.models.train import run_experiments  # noqa: E402

MODE = "overlap_aware"


def main() -> None:
    t0 = time.time()

    print(f"[1/4] Preprocessing (mode={MODE})...", flush=True)
    manual, synthetic, _combined = run_preprocessing(mode=MODE, save=False)
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

    print(f"\nDone in {(time.time() - t0)/60:.1f} min. Outputs in {OVERLAP_RUN_DIR}/")


if __name__ == "__main__":
    main()
