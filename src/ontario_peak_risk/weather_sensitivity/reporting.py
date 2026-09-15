"""Markdown reports generated from actual execution results."""
def write_text(path,text): path.write_text(text,encoding='utf-8')
def design_report(path,cfg):
    ds=', '.join(f'{x:+g}°C' if x else 'Baseline' for x in cfg['analysis']['scenarios_c'])
    write_text(path,f'''# WSA_00 — Weather Sensitivity Design and Scope\n\n## Objective\nEvaluate how frozen Model v1 predictions respond to controlled perturbations of `origin__Temp (°C)` while all other model features remain fixed.\n\n## Scenarios\n{ds}\n\n## Interpretation boundary\nThis is a model sensitivity / what-if analysis, not a causal estimate and not a future target-hour weather experiment. Model v1 only contains forecast-origin weather. Models are not retrained.\n\n## OOD policy\nScenario temperatures are labelled NORMAL, CAUTION, or OUT_OF_RANGE using the model-development feature dataset. Predictions may be retained with warnings; they are not treated as equally reliable outside training support.\n''')
