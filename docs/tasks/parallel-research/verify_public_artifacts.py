"""Recheck complete public runs; invoke from repository root with task slugs.

Example: .venv/bin/python docs/tasks/parallel-research/verify_public_artifacts.py regression classification
Outputs JSON; requires local runs/parallel-public and original public data.
"""

import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib

sys.path.insert(0, str(Path.cwd()))
from automl_agent.contracts import TaskSpec
from automl_agent.evaluation import score_trial, validate_submission
from automl_agent.budgeting import plan_finalization
from automl_agent.contracts import ExperimentResult

root = Path.cwd()
result = {'date': '2026-10-19', 'provider': 'ollama', 'runs': {}}
for name in sys.argv[1:]:
    run = root/'runs'/'parallel-public'/name
    read = lambda name: json.loads((run/name).read_text())
    report = read('report.json')
    task = TaskSpec.from_dict(read('task.json'))
    trials = read('experiments.json')
    if isinstance(trials,dict):
        trials=trials['trials']
    split=read('split_manifest.json')
    events=[{**e, **e.get('payload',{})} for e in map(json.loads, (run/'events.jsonl').read_text().splitlines())]
    processes=[]
    scored=[]
    for trial in trials:
        if trial['status']=='ok':
            metric=score_trial(task, split, Path(trial['artifact_dir']))[task.metric]
            assert np.isclose(metric,trial['metric'],rtol=1e-12), (name,trial['trial_id'])
            scored.append({'trial_id':trial['trial_id'],'metric':metric,'origin':trial.get('origin'),'model_details':trial.get('model_details')})
        process_path=Path(trial['artifact_dir'])/'process.json'
        if process_path.exists():
            proc=json.loads(process_path.read_text())
            if proc.get('started_at') is not None:
                assert proc['n_jobs']==4
                processes.append({'trial_id':trial['trial_id'],**{k:proc[k] for k in ['started_at','finished_at','pid','n_jobs','status']}})
    overlaps=[]
    for i,a in enumerate(processes):
        for b in processes[i+1:]:
            delta=min(a['finished_at'],b['finished_at'])-max(a['started_at'],b['started_at'])
            if delta>0: overlaps.append({'trials':[a['trial_id'],b['trial_id']],'seconds':delta})
    assert overlaps, name
    boundaries=sorted([(p['started_at'],1,p['n_jobs']) for p in processes]+[(p['finished_at'],-1,-p['n_jobs']) for p in processes])
    concurrent=cpu=maximum=maximum_cpu=0
    for t,n,c in boundaries:
        concurrent+=n; cpu+=c
        maximum=max(maximum,concurrent); maximum_cpu=max(maximum_cpu,cpu)
    assert maximum<=2 and maximum_cpu<=8
    submission=Path(report['submission_path'])
    csv_check=validate_submission(task,submission)
    assert report['final_fit']['status']=='ok', report
    best=next(t for t in trials if t['trial_id']==report['selected_trial'])
    ordered=sorted([t for t in trials if t['status']=='ok' and t['metric'] is not None],key=lambda t:int(t['trial_id'].split('_')[1]))
    chosen=(max if task.maximize else min)(ordered,key=lambda t:t['metric'])
    assert chosen['trial_id']==report['selected_trial']
    best_result=ExperimentResult(**best)
    train_rows=len(pd.read_csv(task.train_path))
    calculated=plan_finalization(total_seconds=read('config.json')['time_budget'],validation_fraction=task.validation_fraction,best=best_result,train_rows=train_rows)
    assert calculated==report['reserve_plan']
    model_path=Path(report['final_fit']['model_path'])
    model=joblib.load(model_path)
    params=model.get_params(deep=True)
    thread_parameters={k:v for k,v in params.items() if k.split('__')[-1] in ['n_jobs','thread_count','nthread','num_threads']}
    assert thread_parameters
    assert all(v is None or 1<=int(v)<=4 for v in thread_parameters.values()),thread_parameters
    data=pd.read_csv(task.test_path,low_memory=False)
    pred=np.asarray(model.predict(data[task.feature_columns].iloc[:50])).reshape(-1)
    actual=pd.read_csv(submission)[task.target].iloc[:50].to_numpy()
    if task.task_type=='regression': assert np.allclose(pred.astype(float),actual.astype(float))
    else: assert np.array_equal(pred.astype(str),actual.astype(str))
    final_event=next(e for e in events if e['event']=='final_fit_started')
    final_process=json.loads((Path(report['final_fit']['artifact_dir'])/'process.json').read_text())
    for proc in processes:
        assert proc['finished_at']<=final_process['started_at']
    result['runs'][name]={
        'run_dir':str(run),'config':read('config.json'),'status':report['status'],'selected_trial':report['selected_trial'],
        'metric':report['metric'],'best_metric':report['best_metric'],'validation':csv_check,'elapsed_seconds':report.get('elapsed_seconds'),
        'family_comparison':read('family_comparison.json'),'rescored':scored,
        'trial_summary':[{k:t.get(k) for k in ['trial_id','status','metric','seconds','origin','hypothesis','code_hash']} for t in trials],
        'llm_tool_calls':[e['name'] for e in events if e['event']=='tool_call'],
        'processes':processes,'overlaps':overlaps,'max_concurrent_workers':maximum,'max_allocated_cpus':maximum_cpu,
        'reserve_plan':report['reserve_plan'],
        'reserve_events':[e for e in events if e['event']=='reserve_updated'],
        'final_fit':report['final_fit'],'final_process':final_process,
        'final_fit_timeout_seconds':final_event.get('timeout_seconds'),
        'model_reload_predict_matches':True,'model_thread_parameters':thread_parameters,
        'final_model_sha256':hashlib.sha256(model_path.read_bytes()).hexdigest(),'csv_sha256':hashlib.sha256(submission.read_bytes()).hexdigest(),
        'events_sha256':hashlib.sha256((run/'events.jsonl').read_bytes()).hexdigest(),
        'llm_calls':report.get('llm_calls'),'llm_trials':report.get('llm_trials'),
        'successful_llm_trials':sum(t['status']=='ok' and not t['is_baseline'] for t in trials)
    }
print(json.dumps(result,ensure_ascii=False,indent=2))
