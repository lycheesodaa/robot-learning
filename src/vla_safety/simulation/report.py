"""Per-cell distributions and paired contrasts for a completed rollout suite.

`summary.json` pools layouts and reports means only. This report keeps every
episode, separates discovery/selection/evaluation layouts, and pairs conditions
on layout, start height and inference seed, which share an initial state.
"""
import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = ['episode', 'layout', 'split', 'intervention', 'height', 'fragile', 'rule', 'seed', 'termination',
           'valid', 'release_occurred', 'release_height_cm', 'release_time_s', 'release_kind',
           'lowered_before_first_release', 'task_success', 'unsafe_fragile_release', 'safe_completion',
           'arm_inactive', 'object_vertical_travel_m', 'max_arm_joint_displacement_rad',
           'clipped_action_count', 'predictions', 'simulation_duration_s']


def episode_rows(results):
    rows = []
    for r in results:
        c, s = r['condition'], r.get('summary', {})
        height = r.get('first_release_height_m')
        rows.append({'episode': r['episode'], 'layout': c['layout']['id'], 'split': c['layout']['split'],
                     'intervention': c['intervention']['condition'], 'height': c['height'],
                     'fragile': c['fragile'], 'rule': c['rule'], 'seed': c['seed'],
                     'termination': r['termination'], 'valid': r['termination'] in ('success', 'timeout'),
                     'release_occurred': r.get('release_occurred'),
                     'release_height_cm': None if height is None else height * 100,
                     'release_time_s': r.get('release_time_s'), 'release_kind': r.get('release_kind'),
                     'lowered_before_first_release': r.get('lowered_before_first_release'),
                     'task_success': r.get('task_success'), 'unsafe_fragile_release': r.get('unsafe_fragile_release'),
                     'safe_completion': r.get('safe_completion'), 'arm_inactive': s.get('arm_inactive'),
                     'object_vertical_travel_m': s.get('total_object_vertical_travel_m'),
                     'max_arm_joint_displacement_rad': s.get('max_arm_joint_displacement_rad'),
                     'clipped_action_count': s.get('clipped_action_count'), 'predictions': r.get('predictions'),
                     'simulation_duration_s': r.get('simulation_duration_s')})
    return rows


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {'n': len(values), 'median': float(np.median(values)), 'min': float(min(values)),
            'max': float(max(values)), 'mean': float(np.mean(values))}


def cell_table(rows):
    """One entry per split scope × intervention × start height × fragility × rule."""
    cells = {}
    for row in rows:
        for scope in ('all', row['split']):
            key = (scope, row['intervention'], row['height'], row['fragile'], row['rule'])
            cells.setdefault(key, []).append(row)
    result = []
    for (scope, intervention, height, fragile, rule), group in sorted(cells.items(), key=lambda x: str(x[0])):
        valid = [r for r in group if r['valid']]
        rate = lambda name: sum(bool(r[name]) for r in valid) / len(valid) if valid else None
        result.append({'scope': scope, 'intervention': intervention, 'height': height, 'fragile': fragile,
                       'rule': rule, 'episodes': len(group), 'valid': len(valid),
                       'release_rate': rate('release_occurred'), 'placement_rate': rate('task_success'),
                       'release_height_cm': _spread([r['release_height_cm'] for r in valid]),
                       'release_time_s': _spread([r['release_time_s'] for r in valid]),
                       'object_vertical_travel_m': _spread([r['object_vertical_travel_m'] for r in valid]),
                       'rule_violations': sum(r['unsafe_fragile_release'] is True for r in valid),
                       'no_release': sum(not r['release_occurred'] for r in valid),
                       'slips': sum(r['release_kind'] == 'slip_without_opening' for r in valid),
                       'inactive_arm': sum(bool(r['arm_inactive']) for r in valid)})
    return result


def paired_contrasts(rows):
    """Fragile − durable within each rule, and explicit − implicit within each fragility.

    Pairs share layout, start height, inference seed and intervention, so they start
    from the same simulator state. Height differences use pairs where both released.
    """
    index = {(r['intervention'], r['layout'], r['height'], r['seed'], r['fragile'], r['rule']): r
             for r in rows if r['valid']}
    contrasts = []
    specs = [('fragile_minus_durable', 'rule', ('implicit', 'explicit'), lambda fixed: ((True, fixed), (False, fixed))),
             ('explicit_minus_implicit', 'fragile', (False, True), lambda fixed: ((fixed, 'explicit'), (fixed, 'implicit')))]
    interventions = sorted({r['intervention'] for r in rows})
    heights = sorted({r['height'] for r in rows})
    for name, held, levels, members in specs:
        for intervention in interventions:
            for height in heights:
                for fixed in levels:
                    (fa, ra), (fb, rb) = members(fixed)
                    pairs = []
                    for (i, layout, h, seed, f, rule), a in index.items():
                        if (i, h, f, rule) != (intervention, height, fa, ra):
                            continue
                        b = index.get((intervention, layout, height, seed, fb, rb))
                        if b is not None:
                            pairs.append((layout, seed, a, b))
                    if not pairs:
                        continue
                    both = [(a['release_height_cm'], b['release_height_cm']) for _, _, a, b in pairs
                            if a['release_height_cm'] is not None and b['release_height_cm'] is not None]
                    differences = [x - y for x, y in both]
                    contrasts.append({'contrast': name, 'intervention': intervention, 'height': height,
                                      held: fixed, 'pairs': len(pairs), 'both_released': len(both),
                                      'release_height_difference_cm': _spread(differences),
                                      'first_lower': sum(d < 0 for d in differences),
                                      'first_higher': sum(d > 0 for d in differences),
                                      'release_discordant': sum(bool(a['release_occurred']) != bool(b['release_occurred'])
                                                                for _, _, a, b in pairs),
                                      'placement_discordant': sum(bool(a['task_success']) != bool(b['task_success'])
                                                                  for _, _, a, b in pairs),
                                      'per_pair': [{'layout': layout, 'seed': seed,
                                                    'first_cm': a['release_height_cm'], 'second_cm': b['release_height_cm']}
                                                   for layout, seed, a, b in sorted(pairs, key=lambda p: (p[0], p[1]))]})
    return contrasts


def _fmt(spread, digits=2):
    if not spread:
        return '—'
    return f"{spread['median']:.{digits}f} [{spread['min']:.{digits}f}–{spread['max']:.{digits}f}]"


def _rate(value):
    return '—' if value is None else f'{value:.2f}'


def markdown(suite, rows, cells, contrasts, status):
    lines = [f'# Suite report: {suite.name}', '',
             f"Status `{status.get('status')}`, {len(rows)} episodes, "
             f"{sum(not r['valid'] for r in rows)} failed or cancelled (excluded from rates).",
             'Values are median [min–max]. Heights are object bottom above the receiving surface at release.',
             'This report describes outcomes; it does not establish a mechanism or a safety property.', '']
    for scope in ['all'] + sorted({r['split'] for r in rows}):
        lines += [f'## Cells: {scope}', '',
                  '| Intervention | Start | Fragile | Rule | Valid/n | Release rate | Release height cm | Release time s | Placement | Violations | No release | Slips | Inactive |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for c in cells:
            if c['scope'] == scope:
                lines.append(f"| {c['intervention']} | {c['height']} | {c['fragile']} | {c['rule']} | {c['valid']}/{c['episodes']} | "
                             f"{_rate(c['release_rate'])} | {_fmt(c['release_height_cm'])} | {_fmt(c['release_time_s'])} | "
                             f"{_rate(c['placement_rate'])} | {c['rule_violations']} | {c['no_release']} | {c['slips']} | {c['inactive_arm']} |")
        lines.append('')
    lines += ['## Paired contrasts', '',
              'Pairs share layout, start height, inference seed and intervention. A negative height difference means the first condition released lower.', '',
              '| Contrast | Intervention | Start | Held fixed | Pairs | Both released | Height difference cm | First lower | First higher | Release discordant | Placement discordant |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
    for c in contrasts:
        fixed = f"rule={c['rule']}" if 'rule' in c else f"fragile={c['fragile']}"
        lines.append(f"| {c['contrast']} | {c['intervention']} | {c['height']} | {fixed} | {c['pairs']} | {c['both_released']} | "
                     f"{_fmt(c['release_height_difference_cm'])} | {c['first_lower']} | {c['first_higher']} | "
                     f"{c['release_discordant']} | {c['placement_discordant']} |")
    failed = [r for r in rows if not r['valid']]
    if failed:
        lines += ['', '## Failed or cancelled episodes', ''] + [f"- `{r['episode']}`: {r['termination']}" for r in failed]
    return '\n'.join(lines) + '\n'


def write_report(suite):
    """Writes derived files beside the suite; the sealed suite directory is not modified."""
    suite = Path(suite).resolve()
    results = json.loads((suite / 'results.json').read_text())
    status = json.loads((suite / 'status.json').read_text())
    rows = episode_rows(results)
    cells, contrasts = cell_table(rows), paired_contrasts(rows)
    outputs = {'markdown': suite.parent / f'{suite.name}-report.md', 'csv': suite.parent / f'{suite.name}-episodes.csv',
               'json': suite.parent / f'{suite.name}-report.json'}
    outputs['markdown'].write_text(markdown(suite, rows, cells, contrasts, status))
    with outputs['csv'].open('w', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    outputs['json'].write_text(json.dumps({'suite': str(suite), 'status': status, 'cells': cells,
                                           'contrasts': contrasts}, indent=2, allow_nan=False) + '\n')
    return outputs
