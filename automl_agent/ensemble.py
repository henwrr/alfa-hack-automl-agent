"""Host selection of OOF postprocessing, with explicit costs and artifacts."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import _read_probability_artifacts
from .postprocessing import apply_postprocessing, select_postprocessing


def _aligned(p, axis, classes):
    if len(axis) != len(classes) or set(axis) != set(classes):
        raise ValueError('Probability class axis differs from the task')
    if p.ndim != 2 or p.shape[1] != len(axis) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError('Invalid probability artifacts')
    return p[:, [axis.index(c) for c in classes]]


def ensemble_shortlist(compatible):
    """Best-first, diverse candidates; shared with final-fit reservation."""
    candidates, seen = [], set()
    for result in compatible:
        family = (result.recipe or {}).get('family', result.model_details.get('class', result.trial_id))
        if family not in seen:
            candidates.append(result); seen.add(family)
        if len(candidates) == 3:
            break
    for result in compatible:
        if len(candidates) >= 3:
            break
        if result not in candidates:
            candidates.append(result)
    return candidates


def choose_postprocessing(task, split, results, run_dir, *, remaining_seconds, overhead_seconds):
    started = time.monotonic()
    if split.get('mode') != 'evaluate_cv':
        return {'accepted':False, 'reason':'legacy_holdout_has_no_oof'}
    compatible = [r for r in results if r.status == 'ok' and r.metric is not None
                  and r.model_details.get('training',{}).get('split_fingerprint') == split['split_fingerprint']]
    compatible.sort(key=lambda r:(-r.metric if task.maximize else r.metric, r.trial_id))
    if not compatible:
        return {'accepted':False, 'reason':'no_compatible_oof'}
    candidates = ensemble_shortlist(compatible)
    best = compatible[0]
    classes = list(getattr(task, 'classes', []))
    y = pd.read_csv(split['validation_target'])[task.target].to_numpy()
    predictions, probabilities, test_predictions, test_probabilities = [], [], [], []
    folds = None
    usable = []
    for result in candidates:
        directory = Path(result.artifact_dir)
        try:
            valid = pd.read_csv(directory/'valid_predictions.csv')[task.target].to_numpy()
            test = pd.read_csv(directory/'test_predictions.csv')[task.target].to_numpy()
            ids = np.load(directory/'oof_fold_ids.npy', allow_pickle=False)
            if len(valid) != len(y) or len(ids) != len(y) or (folds is not None and not np.array_equal(ids, folds)):
                raise ValueError('OOF fold axes differ')
            if task.task_type == 'classification':
                artifacts = _read_probability_artifacts(directory)
                if artifacts is None:
                    raise ValueError('Classification postprocessing requires probability artifacts')
                p, axis = artifacts
                pt = np.load(directory/'test_probabilities.npy', allow_pickle=False)
                vp, tp = _aligned(p, axis, classes), _aligned(pt, axis, classes)
                if len(vp) != len(y) or len(tp) != len(test):
                    raise ValueError('Probability row counts differ')
                probabilities.append(vp)
                test_probabilities.append(tp)
            predictions.append(valid); test_predictions.append(test); usable.append(result); folds = ids
        except (ValueError, KeyError, OSError):
            if result is best:
                return {'accepted':False, 'reason':'best_oof_artifacts_unavailable'}
    if not usable or usable[0] is not best:
        return {'accepted':False, 'reason':'best_oof_artifacts_unavailable'}
    plan = select_postprocessing(y=y, predictions=predictions,
                probabilities=probabilities if task.task_type == 'classification' else None,
                classes=classes, fold_ids=folds, metric=task.metric, maximize=task.maximize,
                temporal=split.get('strategy') == 'forward_group')
    plan['classes'] = classes
    plan['member_trial_ids'] = [r.trial_id for r in usable]
    plan['split_fingerprint'] = split['split_fingerprint']
    plan.setdefault('warnings', []).append('Meta-fold assessment reuses adaptively selected base OOF models; this is not nested CV or an independent test score.')
    active = [i for i,w in enumerate(plan['weights']) if w > 0]
    estimate = sum(usable[i].model_details.get('training',{}).get('estimated_refit_seconds') or usable[i].seconds for i in active)
    plan['estimated_fit_seconds'] = float(estimate*1.5)
    plan['reason'] = 'meta_improvement' if plan['accepted'] else 'no_meta_improvement'
    if plan['accepted'] and estimate*1.5 + overhead_seconds + time.monotonic()-started >= remaining_seconds:
        plan.update(accepted=False, reason='insufficient_refit_time')
        plan['warnings'].append('Недостаточно времени на полный refit всех участников; сохранена лучшая одиночная модель.')
    if plan['accepted']:
        prediction, _ = apply_postprocessing(test_predictions,
            test_probabilities if task.task_type == 'classification' else None,
            classes, plan['weights'], plan['class_multipliers'], task.task_type)
        from .worker import _atomic_csv
        path = Path(run_dir)/'postprocessed_test_predictions.csv'
        _atomic_csv(pd.DataFrame({task.target:prediction}), path)
        plan['prediction_path'] = str(path)
    plan['seconds'] = time.monotonic()-started
    return plan
