#!/usr/bin/env python3
"""Independently rescore a completed run and verify its serialized final model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def verify_run(directory):
    import joblib
    import numpy as np
    import pandas as pd
    from automl_agent.contracts import TaskSpec
    from automl_agent.evaluation import score_trial, validate_submission
    from automl_agent.worker import _load_features

    directory = Path(directory).resolve()
    read = lambda name: json.loads((directory/name).read_text(encoding='utf-8'))
    report, split, config = read('report.json'), read('split_manifest.json'), read('config.json')
    task = TaskSpec.from_dict(read('task.json'))
    validation = validate_submission(task, Path(report['submission_path']))
    rescored, intervals = [], []
    for trial in report['trials']:
        artifact = Path(trial['artifact_dir'])
        if trial['status'] == 'ok':
            metric = score_trial(task, split, artifact)[task.metric]
            if not math.isclose(metric, trial['metric'], rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Stored metric differs from host rescore: ' + trial['trial_id'])
            if hashlib.sha256((artifact/'code.py').read_bytes()).hexdigest() != trial['code_hash']:
                raise ValueError('Candidate source checksum changed')
            rescored.append({'trial_id':trial['trial_id'], 'metric':metric})
        process_path = artifact/'process.json'
        if process_path.exists():
            process=json.loads(process_path.read_text())
            if process.get('started_at') is not None and process.get('finished_at') is not None:
                intervals.extend([(process['started_at'], 1, process.get('n_jobs') or 0),
                                  (process['finished_at'], -1, -(process.get('n_jobs') or 0))])
    active=cpu=peak=peak_cpu=0
    for _, delta, threads in sorted(intervals):
        active += delta; cpu += threads; peak=max(peak,active);peak_cpu=max(peak_cpu,cpu)
    if peak > config['parallel_trials'] or peak_cpu > 8:
        raise ValueError('Measured worker concurrency exceeds configured limits')
    final = report['final_fit']
    if final['status'] != 'ok' or not final.get('model_path'):
        raise ValueError('Final refit/model was not successful')
    model=joblib.load(final['model_path'])
    X=_load_features(split['test_features'],task.feature_columns,set(task.categorical_columns))
    prediction=np.asarray(model.predict(X)).reshape(-1)
    submitted=pd.read_csv(report['submission_path'])[task.target].to_numpy()
    if task.task_type == 'regression':
        np.testing.assert_allclose(prediction.astype(float),submitted.astype(float),rtol=1e-10,atol=1e-10)
    else:
        np.testing.assert_array_equal(prediction,submitted)
    details=final['model_details']
    train_rows=len(pd.read_csv(split['full_train_target']))
    if details.get('training',{}).get('train_rows') != train_rows:
        raise ValueError('Final refit did not record all train rows')
    plan=report.get('postprocessing',{})
    if plan.get('accepted'):
        members=details.get('members',[])
        expected=[trial_id for trial_id,weight in zip(plan['member_trial_ids'],plan['weights']) if weight > 0]
        if [member['trial_id'] for member in members] != expected:
            raise ValueError('Final ensemble members do not match the selected plan')
        for member in members:
            trial=next(row for row in report['trials'] if row['trial_id']==member['trial_id'])
            expected_iterations=trial['model_details']['training']['final_iterations']
            if expected_iterations and member['training']['best_iteration'] != expected_iterations:
                raise ValueError('Final member ignored selected CV iteration count')
            if member['training']['train_rows'] != train_rows:
                raise ValueError('Ensemble member did not use full train')
    summary={'run_dir':str(directory), 'status':report['status'], 'stop_reason':report['stop_reason'],
             'metric':task.metric, 'best_metric':report['best_metric'], 'selected_trial':report['selected_trial'],
             'llm_calls':report['llm_calls'],'llm_trials':report['llm_trials'],'total_tokens':report['total_tokens'],
             'elapsed_seconds':report['elapsed_seconds'],'validation_plan':report.get('validation_plan'),
             'rescored_trials':rescored,'trial_statuses':{row['trial_id']:row['status'] for row in report['trials']},
             'postprocessing':plan, 'submission':validation, 'final_fit_status':final['status'],
             'final_model_reload_matches_submission':True, 'max_parallel_workers':peak,'max_allocated_cpu':peak_cpu,
             'submission_sha256':hashlib.sha256(Path(report['submission_path']).read_bytes()).hexdigest()}
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    summary=verify_run(args.run_dir)
    encoded=json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(encoded,encoding='utf-8')
    print(encoded,end='')


if __name__=='__main__':
    main()
