"""
Path utilities for storing/retrieving per-checkpoint evaluation metric files.

Layout:
    /notebooks/experiment/codes/metrics/
        circo/        <run_id>__<ckpt_tag>.json
        fashion/      (reserved for future per-checkpoint Fashion JSONs)
        bootstrap/    aggregated bootstrap result files

Naming scheme:
    For checkpoint <root>/<run_id>/<file>:
      - If <file> is 'checkpoint.pt' (i.e. nested as <run>/epoch_NN/checkpoint.pt):
            run_id   = grandparent dir name
            ckpt_tag = parent dir name (e.g. 'epoch_01')
      - Otherwise (e.g. 'epoch_NN_step_SSSSS.pt'):
            run_id   = parent dir name
            ckpt_tag = file stem (without .pt)

    Output filename: <run_id>__<ckpt_tag>.json
    Two underscores '__' as separator (run names may contain single '_').
"""
from pathlib import Path


METRICS_ROOT = Path('/notebooks/experiment/codes/metrics')


def derive_run_id_and_ckpt_tag(checkpoint_path):
    """
    Returns: (run_id, ckpt_tag) — both strings, safe to use in filenames.
    
    Examples:
        <root>/base_hd256.../epoch_01/checkpoint.pt
            → ('base_hd256...', 'epoch_01')
        <root>/base_hd256.../epoch_02_step_03000.pt
            → ('base_hd256...', 'epoch_02_step_03000')
        <root>/some_run/final.pt
            → ('some_run', 'final')
    """
    ckpt = Path(checkpoint_path).resolve()
    if ckpt.stem == 'checkpoint':
        # Nested case: <run>/epoch_NN/checkpoint.pt
        return ckpt.parent.parent.name, ckpt.parent.name
    else:
        # Flat case: <run>/<anything>.pt
        return ckpt.parent.name, ckpt.stem


def metric_file_path(checkpoint_path, benchmark):
    """
    Get the canonical metrics file path for a given (checkpoint, benchmark).
    
    benchmark: 'circo' or 'fashion'
    Returns: Path object pointing to where the per-query metrics JSON should live.
    """
    assert benchmark in ('circo', 'fashion'), f'Unknown benchmark {benchmark}'
    run_id, ckpt_tag = derive_run_id_and_ckpt_tag(checkpoint_path)
    return METRICS_ROOT / benchmark / f'{run_id}__{ckpt_tag}.json'


def ensure_metrics_dirs():
    """Create the metrics/ subdirs if they don't exist (idempotent)."""
    for sub in ('circo', 'fashion', 'bootstrap'):
        (METRICS_ROOT / sub).mkdir(parents=True, exist_ok=True)