"""Core plots for scenario interpretation."""
import matplotlib.pyplot as plt

def scenario_lines(df,y,title,ylabel):
    fig,ax=plt.subplots(figsize=(11,6))
    for s,g in df.groupby('scenario',sort=False): ax.plot(g['horizon'],g[y],marker='o',label=s)
    ax.set(title=title,xlabel='Forecast horizon (hours)',ylabel=ylabel); ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); return fig

def heatmap(df,value,title):
    p=df.pivot_table(index='fsa',columns='horizon',values=value,aggfunc='mean')
    fig,ax=plt.subplots(figsize=(12,4)); im=ax.imshow(p.to_numpy(),aspect='auto'); ax.set(title=title,xlabel='Horizon',ylabel='FSA'); ax.set_xticks(range(len(p.columns)),p.columns); ax.set_yticks(range(len(p.index)),p.index); fig.colorbar(im,ax=ax); fig.tight_layout(); return fig
