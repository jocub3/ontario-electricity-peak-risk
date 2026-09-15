"""Shared configuration and paths for Weather Sensitivity Analysis."""
from pathlib import Path
import yaml

def load_config(path='configs/weather_sensitivity.yaml'):
    p=Path(path).resolve()
    with p.open(encoding='utf-8') as f: cfg=yaml.safe_load(f)
    return cfg, p.parent.parent

def resolve(root, value):
    p=Path(value); return p if p.is_absolute() else root/p

def ensure_dirs(cfg, root):
    out={}
    for k in ['outputs_dir','reports_dir','figures_dir','docs_dir']:
        p=resolve(root,cfg['paths'][k]); p.mkdir(parents=True,exist_ok=True); out[k]=p
    return out
