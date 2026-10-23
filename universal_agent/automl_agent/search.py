"""Small reproducible fallback search inspired by rakovskiy.researcher.

These are host proposals, never counted as experiments proposed by an LLM.
The normal recipe validator, same CV, deduplication and deadlines still apply.
"""
import random

from .recipes import normalize_recipe, recipe_fingerprint


_SPACE = {
    'CatBoost': {'depth':[4,5,7,8], 'l2_leaf_reg':[1.,3.,8.,15.], 'learning_rate':[.025,.04,.08]},
    'LightGBM': {'num_leaves':[15,47,63,95], 'min_child_samples':[10,30,60,100], 'reg_lambda':[.1,1.,5.,15.]},
    'XGBoost': {'max_depth':[3,4,5,8], 'min_child_weight':[1.,3.,8.], 'reg_lambda':[.1,1.,5.,15.]},
    'ExtraTrees': {'min_samples_leaf':[1,2,5,10], 'max_features':[.5,.7,1.], 'max_leaf_nodes':[64,256,512]},
    'Linear': {'C':[.1,1.,3.,10.]},
}


def next_search_proposal(results, task_type, maximize, seed, step):
    measured = [r for r in results if r.status == 'ok' and r.recipe and r.metric is not None]
    if not measured:
        return None
    measured.sort(key=lambda r: r.metric, reverse=maximize)
    seen = {recipe_fingerprint(r.recipe, task_type) for r in results if r.recipe}
    rng = random.Random(int(seed) + 104729 * (step+1))
    for attempt in range(64):
        base = measured[0] if attempt % 2 == 0 else rng.choice(measured)
        space = dict(_SPACE[base.recipe['family']])
        if task_type == 'regression' and base.recipe['family'] == 'Linear':
            continue
        if task_type == 'classification' and base.recipe['family'] == 'CatBoost':
            space['auto_class_weights'] = [None, 'SqrtBalanced', 'Balanced']
        key = rng.choice(sorted(space))
        value = rng.choice(space[key])
        recipe = normalize_recipe(base.recipe, task_type)
        recipe['parameters'][key] = value
        try:
            recipe = normalize_recipe(recipe, task_type)
        except ValueError:
            continue
        if recipe_fingerprint(recipe, task_type) not in seen:
            return {'base_trial_id':base.trial_id, 'parameters':{key:value},
                    'hypothesis':f'Воспроизводимый резервный поиск: проверить {key}={value} у {recipe["family"]} на той же валидации.'}
    return None
