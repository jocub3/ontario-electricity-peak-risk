"""Memory-conscious scenario inference: one horizon model at a time."""
import gc, joblib, numpy as np, pandas as pd
from pathlib import Path

def _model_path(root,cfg,task,h):
    art=root/cfg['paths']['artifacts_dir']; m=cfg['models']
    if task=='rf': return art/m['rf_subdir']/m['rf_pattern'].format(horizon=h)
    return art/m['xgb_subdir']/m['xgb_pattern'].format(horizon=h)

def run_sensitivity(scenario_frames, metadata, cfg, root):
    rows=[]; threshold=float(metadata['xgb']['operational_threshold'])
    for h in range(1,int(cfg['analysis']['horizons'])+1):
        rf=joblib.load(_model_path(root,cfg,'rf',h)); xgb=joblib.load(_model_path(root,cfg,'xgb',h))
        rf_f=metadata['rf']['numeric_features']+metadata['rf']['categorical_features']
        xg_f=metadata['xgb']['numeric_features']+metadata['xgb']['categorical_features']
        a=scenario_frames['rf'][h].copy(); b=scenario_frames['xgb'][h].copy()
        keys=['fsa','forecast_origin','target_timestamp','horizon','scenario','temperature_delta_c','baseline_temperature_c','origin__Temp (°C)','temperature_domain_status']
        if not a[keys[:7]].equals(b[keys[:7]]): raise ValueError(f'RF/XGB scenario keys differ at h+{h}.')
        pred=rf.predict(a[rf_f]); prob=xgb.predict_proba(b[xg_f])[:,1]
        z=a[keys].copy(); z['forecast_consumption_kwh']=pred; z['peak_risk_score']=prob; z['peak_alert']=prob>=threshold; rows.append(z)
        del rf,xgb,a,b,pred,prob,z; gc.collect()
    return pd.concat(rows,ignore_index=True).sort_values(['fsa','horizon','temperature_delta_c']).reset_index(drop=True)
