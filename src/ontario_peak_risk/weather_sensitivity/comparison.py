"""Baseline deltas and FSA/horizon summaries."""
import numpy as np, pandas as pd

def add_baseline_deltas(results):
    x=results.copy(); keys=['fsa','horizon']
    base=x.loc[x['temperature_delta_c'].eq(0), keys+['forecast_consumption_kwh','peak_risk_score','peak_alert']].rename(columns={'forecast_consumption_kwh':'baseline_forecast_kwh','peak_risk_score':'baseline_peak_risk_score','peak_alert':'baseline_peak_alert'})
    x=x.merge(base,on=keys,how='left',validate='many_to_one')
    x['forecast_delta_kwh']=x['forecast_consumption_kwh']-x['baseline_forecast_kwh']
    x['forecast_delta_pct']=np.where(x['baseline_forecast_kwh'].ne(0),100*x['forecast_delta_kwh']/x['baseline_forecast_kwh'],np.nan)
    x['peak_risk_delta']=x['peak_risk_score']-x['baseline_peak_risk_score']
    x['alert_changed']=x['peak_alert'].ne(x['baseline_peak_alert'])
    x['alert_transition']=np.where(~x['alert_changed'],'No change',np.where(x['peak_alert'],'No Alert -> Alert','Alert -> No Alert'))
    return x

def summary_by_fsa_scenario(x):
    return x.groupby(['fsa','scenario','temperature_delta_c'],observed=True).agg(mean_abs_forecast_delta_kwh=('forecast_delta_kwh',lambda s:s.abs().mean()),max_abs_forecast_delta_kwh=('forecast_delta_kwh',lambda s:s.abs().max()),mean_abs_forecast_delta_pct=('forecast_delta_pct',lambda s:s.abs().mean()),mean_abs_peak_risk_delta=('peak_risk_delta',lambda s:s.abs().mean()),max_abs_peak_risk_delta=('peak_risk_delta',lambda s:s.abs().max()),alert_changes=('alert_changed','sum'),ood_rows=('temperature_domain_status',lambda s:(s=='OUT_OF_RANGE').sum()),caution_rows=('temperature_domain_status',lambda s:(s=='CAUTION').sum())).reset_index()

def symmetry_table(x):
    keys=['fsa','horizon']; p=x.pivot_table(index=keys,columns='temperature_delta_c',values=['forecast_delta_kwh','peak_risk_delta'],aggfunc='first')
    rows=[]
    for mag in [2.5,5.0]:
        for metric in ['forecast_delta_kwh','peak_risk_delta']:
            pos=p[(metric,mag)]; neg=p[(metric,-mag)]
            rows.append(pd.DataFrame({'fsa':p.index.get_level_values(0),'horizon':p.index.get_level_values(1),'magnitude_c':mag,'metric':metric,'symmetry_residual':pos.to_numpy()+neg.to_numpy()}))
    return pd.concat(rows,ignore_index=True)
