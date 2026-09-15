"""Construct controlled temperature perturbation scenarios."""
import pandas as pd

def scenario_label(delta):
    if float(delta)==0: return 'Baseline'
    return f"{float(delta):+g}C"

def build_scenario_frames(feature_frames, deltas, temperature_feature='origin__Temp (°C)'):
    """Copy frozen RF/XGB frames and alter ONLY origin temperature."""
    out={'rf':{},'xgb':{}}
    for task in out:
        for h, base in feature_frames[task].items():
            parts=[]
            for d in deltas:
                x=base.copy()
                x['baseline_temperature_c']=pd.to_numeric(x[temperature_feature], errors='raise')
                x[temperature_feature]=x['baseline_temperature_c']+float(d)
                x['temperature_delta_c']=float(d)
                x['scenario']=scenario_label(d)
                parts.append(x)
            out[task][int(h)]=pd.concat(parts,ignore_index=True)
    return out

def validate_only_temperature_changed(base, scenario, feature_cols, temp_feature):
    compare=[c for c in feature_cols if c != temp_feature]
    if len(base)*scenario['scenario'].nunique()!=len(scenario):
        raise ValueError('Scenario row count does not match baseline × scenarios.')
    for _, grp in scenario.groupby('scenario',sort=False):
        if not grp[compare].reset_index(drop=True).equals(base[compare].reset_index(drop=True)):
            raise ValueError('A non-temperature model feature changed during scenario construction.')
    return True
