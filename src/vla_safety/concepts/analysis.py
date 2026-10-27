"""Matched contrast analysis and small NumPy L2 logistic probes.

Seeds are averaged within examples. Discovery rankings never use evaluation
labels. Bootstrap units are scenes/sessions and prompt families, not tokens.
"""
import json

import numpy as np

from vla_safety.archive import archive_path, read_events


def load_features(folder):
    examples, features = {}, {}
    for event in read_events(folder):
        if event['kind'] != 'prediction':
            continue
        detail = json.loads(archive_path(folder, event['detail']['path']).read_text())
        if detail['intervention'] is not None:
            continue
        row = detail['input']
        examples[row['id']] = row
        with np.load(archive_path(folder, detail['arrays']['path']), allow_pickle=False) as arrays:
            calls = {}
            for item in detail['components']:
                calls.setdefault(item['component'], []).append(arrays[item['key']].copy())
            for name, values in calls.items():
                features.setdefault(name, {}).setdefault(row['id'], []).append(np.mean(values, axis=0))
    rows = list(examples.values())
    matrices = {}
    for name, values in features.items():
        if len(values) != len(rows):
            continue
        matrices[name] = np.stack([np.mean(values[row['id']], axis=0) for row in rows])
    return rows, matrices


def paired_differences(rows, x, concept, split):
    pairs = {}
    for i, row in enumerate(rows):
        if row['split'] != split:
            continue
        label = row['labels'][concept]
        controls = tuple(sorted((k, v) for k, v in row['labels'].items() if k != concept))
        key = (row.get('pair_id', row['id']), controls)
        pairs.setdefault(key, {}).setdefault(label, []).append(i)
    differences, units = [], []
    for pair in pairs.values():
        if 0 not in pair or 1 not in pair:
            continue
        differences.append(x[pair[1]].mean(axis=0) - x[pair[0]].mean(axis=0))
        row = rows[pair[1][0]]
        units.append((row['group_id'], row['family_id']))
    return np.asarray(differences), units


def factorial_differences(rows, x, factors, split):
    """(F1,H1 - F1,H0) - (F0,H1 - F0,H0), holding other labels fixed."""
    pairs = {}
    for i, row in enumerate(rows):
        if row['split'] != split:
            continue
        controls = tuple(sorted((k, v) for k, v in row['labels'].items() if k not in factors))
        key = (row['pair_id'], controls)
        cell = tuple(row['labels'][k] for k in factors)
        pairs.setdefault(key, {}).setdefault(cell, []).append(i)
    differences, units = [], []
    for cells in pairs.values():
        if set(cells) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            continue
        means = {k: x[v].mean(axis=0) for k, v in cells.items()}
        differences.append(means[(1, 1)] - means[(1, 0)] - means[(0, 1)] + means[(0, 0)])
        row = rows[cells[(1, 1)][0]]
        units.append((row['group_id'], row['family_id']))
    return np.asarray(differences), units


def group_ci(values, units, *, repeats=500, seed=1000):
    """Two-way cluster bootstrap; uncertainty unavailable with one scene/family."""
    groups = sorted({u[0] for u in units})
    families = sorted({u[1] for u in units})
    if len(groups) < 2 or len(families) < 2 or not repeats:
        return None
    g = np.asarray([groups.index(u[0]) for u in units])
    f = np.asarray([families.index(u[1]) for u in units])
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        gw = np.bincount(rng.integers(len(groups), size=len(groups)), minlength=len(groups))
        fw = np.bincount(rng.integers(len(families), size=len(families)), minlength=len(families))
        weights = gw[g] * fw[f]
        if weights.sum():
            means.append(np.average(values, weights=weights))
    return np.quantile(means, [.025, .975]).tolist() if means else None


def metrics(y, score):
    y, score = np.asarray(y), np.asarray(score)
    if len(np.unique(y)) != 2:
        return {'auroc': None, 'balanced_accuracy': None, 'n': len(y)}
    # Mann-Whitney AUROC with average ranks for ties, O(n log n).
    order = np.argsort(score, kind='stable')
    sorted_scores = score[order]
    ranks = np.empty(len(y), dtype=float)
    start = 0
    while start < len(y):
        end = start + 1
        while end < len(y) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    npos, nneg = int(y.sum()), int((1-y).sum())
    auc = (ranks[y == 1].sum() - npos*(npos+1)/2) / (npos*nneg)
    prediction = score >= .5
    ba = ((prediction[y == 1] == 1).mean() + (prediction[y == 0] == 0).mean()) / 2
    return {'auroc': float(auc), 'balanced_accuracy': float(ba), 'n': len(y)}


def fit_probe(x, y, regularization=.01, max_iter=1000):
    """Full-batch L2 logistic regression, training-only scaling, no policy updates.

    Divide standardized features by sqrt(width), making the regularization grid
    comparable across feature counts. A trace bound supplies a stable step size.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(x).all() or set(y.tolist()) != {0., 1.}:
        raise ValueError('Probe training needs finite features and both classes')
    mean, std = x.mean(axis=0), x.std(axis=0)
    scale = np.where(std > 1e-8, std, 1.) * np.sqrt(x.shape[1])
    z = (x - mean) / scale
    z = np.column_stack((z, np.ones(len(z))))
    weights = np.zeros(z.shape[1])
    step = 1 / (.25 * np.sum(z*z)/len(z) + regularization)
    converged = False
    for iteration in range(max_iter):
        probability = 1 / (1 + np.exp(-np.clip(z @ weights, -40, 40)))
        grad = z.T @ (probability - y) / len(y)
        grad[:-1] += regularization * weights[:-1]
        weights -= step * grad
        if np.linalg.norm(grad) < 1e-5:
            converged = True
            break
    return {'mean': mean, 'scale': scale, 'weights': weights[:-1], 'bias': float(weights[-1]),
            'regularization': regularization, 'iterations': iteration + 1, 'converged': converged}


def probe_scores(probe, x):
    logits = (np.asarray(x) - probe['mean']) / probe['scale'] @ probe['weights'] + probe['bias']
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def analyse(folder, concept, *, top_k=32, bootstrap=500, seed=1000, progress=None, interaction=None):
    rows, matrices = load_features(folder)
    if not rows or not matrices:
        raise ValueError('No completed baseline activations to analyse')
    required = interaction or [concept]
    if any(row.get('labels', {}).get(label) not in (0, 1) for row in rows for label in required):
        raise ValueError(f'Every example requires binary labels for {required}')
    def contrasts(x, split):
        return (factorial_differences(rows, x, interaction, split) if interaction else
                paired_differences(rows, x, concept, split))
    split = 'discovery' if any(row['split'] == 'discovery' for row in rows) else 'exploratory'
    eligible = np.array([row['split'] == split for row in rows])
    if not eligible.any():
        raise ValueError('No discovery or exploratory examples')
    rankings, component_stats, ordering = [], [], {}
    for name, x in matrices.items():
        differences, units = contrasts(x, split)
        if not len(differences):
            raise ValueError('Matched positive/negative pairs are required for contrast ranking')
        scale = np.maximum(x[eligible].std(axis=0), 1e-6)
        effect = differences.mean(axis=0) / scale
        order = np.argsort(-np.abs(effect), kind='stable')
        ordering[name] = order
        component_stats.append({'component': name, 'score': float(np.abs(effect[order[:8]]).mean()),
                                'channels': x.shape[1], 'pairs': len(differences)})
        for channel in order[:top_k]:
            values = differences[:, channel] / scale[channel]
            rankings.append({'component': name, 'channel': int(channel), 'effect': float(effect[channel]),
                             'consistency': float(np.mean(np.sign(values) == np.sign(effect[channel]))),
                             'pairs': len(values), 'ci95': group_ci(values, units, repeats=bootstrap, seed=seed)})
        if progress is not None:
            progress(stage='rankings', component=name, completed=len(ordering), total=len(matrices))
    rankings.sort(key=lambda row: -abs(row['effect']))
    component_stats.sort(key=lambda row: -row['score'])
    result = {'concept': concept, 'evidence': 'exploratory' if split == 'exploratory' else 'discovery',
              'ranking_split': split, 'examples': len(rows), 'rankings': rankings,
              'components': component_stats, 'probes': [], 'probe_trials': [],
              'notes': ['Seeds averaged within examples; flow invocations averaged for ranking.',
                        'Channels are screened by association, not proven concept mechanisms.',
                        'Confidence intervals are descriptive and not corrected for neuron selection.',
                        'Prompt evidence does not establish visual recognition or physical safety.']}
    model_arrays = {}
    if interaction:
        result['contrast'] = {'factors': list(interaction), 'formula': '(11 - 10) - (01 - 00)',
                              'other_factors': 'held fixed within matched sets'}
        result['held_out_interaction'] = []
        for name, x in matrices.items():
            differences, units = contrasts(x, 'evaluation')
            if split != 'discovery' or not len(differences):
                continue
            scale = np.maximum(x[eligible].std(axis=0), 1e-6)
            for channel in ordering[name][:top_k]:
                values = differences[:, channel] / scale[channel]
                result['held_out_interaction'].append({
                    'component': name, 'channel': int(channel), 'effect': float(values.mean()),
                    'ci95': group_ci(values, units, repeats=bootstrap, seed=seed), 'matched_sets': len(values)})
        if result['held_out_interaction']:
            result['evidence'] = 'held_out_interaction'
        result['notes'].append('Interaction contrasts reject purely additive factor responses; they do not demonstrate a safety mechanism.')
        return result, model_arrays
    indices = {s: np.array([i for i, row in enumerate(rows) if row['split'] == s], dtype=int)
               for s in ('discovery', 'selection', 'evaluation')}
    labels = np.array([row['labels'][concept] for row in rows])
    if any(len(np.unique(labels[indices[s]])) != 2 for s in indices):
        result['notes'].append('Held-out probes unavailable: discovery, selection and evaluation each need both labels.')
        return result, model_arrays
    # Screen three components on discovery only; selection chooses group size/lambda.
    for number, component in enumerate(component_stats[:3]):
        name = component['component']
        x = matrices[name]
        order = ordering[name]
        best = None
        for size in sorted({min(k, x.shape[1]) for k in (8, 32, 128, x.shape[1])}):
            selected = order[:size]
            for reg in (.001, .01, .1):
                p = fit_probe(x[indices['discovery']][:, selected], labels[indices['discovery']], reg)
                scores = probe_scores(p, x[indices['selection']][:, selected])
                m = metrics(labels[indices['selection']], scores)
                result['probe_trials'].append({'component': name, 'size': size, 'regularization': reg,
                                                'selection': m, 'converged': p['converged'],
                                                'iterations': p['iterations']})
                if best is None or m['auroc'] > best[0]:
                    best = (m['auroc'], selected, p, m)
        _, selected, p, selection_metrics = best
        scores = probe_scores(p, x[indices['evaluation']][:, selected])
        evaluation_rows = [rows[i] for i in indices['evaluation']]
        family_metrics = {}
        for family in sorted({row['family_id'] for row in evaluation_rows}):
            mask = np.array([row['family_id'] == family for row in evaluation_rows])
            family_metrics[family] = metrics(labels[indices['evaluation']][mask], scores[mask])
        # Frozen probe evaluated against a pair-preserving null: flip whole quartets.
        rng = np.random.default_rng(seed)
        null_aucs = []
        pair_keys = [row.get('pair_id', row['id']) for row in evaluation_rows]
        groups = sorted(set(pair_keys))
        for _ in range(100):
            flips = dict(zip(groups, rng.integers(2, size=len(groups)).tolist()))
            shuffled = np.array([labels[i] ^ flips[key] for i, key in zip(indices['evaluation'], pair_keys)])
            value = metrics(shuffled, scores)['auroc']
            if value is not None:
                null_aucs.append(value)
        evaluation_diffs, _ = paired_differences(rows, x, concept, 'evaluation')
        held_order = np.argsort(-np.abs(evaluation_diffs.mean(axis=0)), kind='stable') if len(evaluation_diffs) else []
        k = min(top_k, x.shape[1])
        overlap = len(set(order[:k]) & set(held_order[:k])) / k if len(held_order) else None
        result['probes'].append({'component': name, 'neurons': selected.tolist(),
                                 'regularization': p['regularization'], 'converged': p['converged'],
                                 'selection': selection_metrics, 'evaluation': metrics(labels[indices['evaluation']], scores),
                                 'families': family_metrics, 'null_auroc95': np.quantile(null_aucs, [.025, .975]).tolist() if null_aucs else None,
                                 'evaluation_top_k_overlap': overlap, 'array_prefix': f'probe{number}'})
        for key in ('mean', 'scale', 'weights', 'bias'):
            model_arrays[f'probe{number}_{key}'] = np.asarray(p[key])
        model_arrays[f'probe{number}_neurons'] = selected
        if progress is not None:
            progress(stage='probes', component=name, completed=number + 1, total=min(3, len(component_stats)))
    result['evidence'] = 'held_out_association'
    return result, model_arrays
