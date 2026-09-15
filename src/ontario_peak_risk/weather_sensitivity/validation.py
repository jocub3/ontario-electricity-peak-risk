"""Model-development temperature support and scenario OOD checks."""
import pandas as pd

def temperature_reference(feature_dataset, feature='origin__Temp (°C)', q=(.01,.99)):
    s=pd.to_numeric(feature_dataset[feature],errors='coerce').dropna()
    if s.empty: raise ValueError(f'No usable values for {feature}.')
    return {'min':float(s.min()),'q_low':float(s.quantile(q[0])),'q_high':float(s.quantile(q[1])),'max':float(s.max()),'n':int(s.size)}

def domain_status(value, ref):
    v=float(value)
    if v < ref['min'] or v > ref['max']: return 'OUT_OF_RANGE'
    if v < ref['q_low'] or v > ref['q_high']: return 'CAUTION'
    return 'NORMAL'

def add_domain_status(df, ref, temperature_feature='origin__Temp (°C)'):
    x=df.copy(); x['temperature_domain_status']=x[temperature_feature].map(lambda v: domain_status(v,ref)); return x
