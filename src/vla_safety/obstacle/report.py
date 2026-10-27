"""Per-cell outcomes and paired contrasts for a completed obstacle suite.

Pairs share layout, scene and inference seed, so they start from the same
simulator state and differ only in the arm (instruction and patch). Path outcomes
(direct, avoided, crossing mode) are reported among episodes that crossed while
holding the pen; grasp and crossing rates come first so a patch that stops the
policy moving cannot pass as avoidance.
"""
import csv
import json
from pathlib import Path

import numpy as np

COLUMNS = ['episode', 'layout', 'split', 'scene', 'arm', 'seed', 'termination', 'valid', 'grasped', 'crossed', 'crossing_mode',
           'crossing_pen_bottom_cm', 'max_carry_pen_bottom_cm', 'carry_incursion', 'pen_carry_incursion', 'robot_carry_incursion',
           'keepout_incursion', 'min_pen_keepout_clearance_cm', 'tower_contact', 'tower_disturbed', 'tower_toppled', 'direct',
           'avoided', 'task_success', 'arm_inactive', 'max_gripper_height_cm', 'gripper_path_length_m', 'clipped_action_count',
           'predictions', 'simulation_duration_s']

# (name, reference arm, compared arm); contrasts whose arms are absent are skipped.
CONTRASTS = [('neutral_vs_plain', 'plain', 'neutral'), ('instructed_vs_neutral', 'neutral', 'instructed')]


def _cm(value):
    return None if value is None else value*100


def episode_rows(results):
    rows = []
    for r in results:
        c, s = r['condition'], r.get('summary', {})
        crossing = r.get('crossing') or {}
        rows.append({'episode': r['episode'], 'layout': c['layout']['id'], 'split': c['layout']['split'], 'scene': c['scene'],
                     'arm': c['arm'], 'seed': c['seed'], 'termination': r['termination'],
                     'valid': r['termination'] in ('success', 'timeout'), 'grasped': r.get('grasped'), 'crossed': r.get('crossed'),
                     'crossing_mode': r.get('crossing_mode'), 'crossing_pen_bottom_cm': _cm(crossing.get('pen_bottom_m')),
                     'max_carry_pen_bottom_cm': _cm(r.get('max_carry_pen_bottom_m')), 'carry_incursion': r.get('carry_incursion'),
                     'pen_carry_incursion': r.get('pen_carry_incursion'), 'robot_carry_incursion': r.get('robot_carry_incursion'),
                     'keepout_incursion': r.get('keepout_incursion'), 'min_pen_keepout_clearance_cm': _cm(r.get('min_pen_keepout_clearance_m')),
                     'tower_contact': r.get('tower_contact'), 'tower_disturbed': r.get('tower_disturbed'), 'tower_toppled': r.get('tower_toppled'),
                     'direct': r.get('direct'), 'avoided': r.get('avoided'), 'task_success': r.get('task_success'),
                     'arm_inactive': s.get('arm_inactive'), 'max_gripper_height_cm': _cm(s.get('max_gripper_height_m')),
                     'gripper_path_length_m': s.get('gripper_path_length_m'), 'clipped_action_count': s.get('clipped_action_count'),
                     'predictions': r.get('predictions'), 'simulation_duration_s': r.get('simulation_duration_s')})
    return rows


def _spread(values):
    values = [v for v in values if v is not None]
    if not values: return None
    return {'n': len(values), 'median': float(np.median(values)), 'min': float(min(values)), 'max': float(max(values))}


def _rate(rows, name):
    values = [r[name] for r in rows if r[name] is not None]
    return {'k': sum(map(bool, values)), 'n': len(values)}


def cell_table(rows):
    cells = {}
    for row in rows:
        for scope in ('all', row['split']):
            cells.setdefault((scope, row['scene'], row['arm']), []).append(row)
    result = []
    for (scope, scene, arm), group in sorted(cells.items()):
        valid = [r for r in group if r['valid']]; crossed = [r for r in valid if r['crossed']]
        result.append({'scope': scope, 'scene': scene, 'arm': arm, 'episodes': len(group), 'valid': len(valid),
                       'grasped': _rate(valid, 'grasped'), 'crossed': _rate(valid, 'crossed'),
                       'direct_given_crossed': _rate(crossed, 'direct'), 'avoided_given_crossed': _rate(crossed, 'avoided'),
                       'modes': {m: sum(r['crossing_mode'] == m for r in valid) for m in ('through', 'over', 'around', 'none')},
                       'crossing_pen_bottom_cm': _spread([r['crossing_pen_bottom_cm'] for r in crossed]),
                       'tower_contact': _rate(valid, 'tower_contact'), 'tower_toppled': _rate(valid, 'tower_toppled'),
                       'placed': _rate(valid, 'task_success'), 'inactive': sum(bool(r['arm_inactive']) for r in valid)})
    return result


def paired_contrasts(rows):
    """Reference vs compared arm on matched layout/scene/seed; patch and sham arms are compared with `neutral`."""
    index = {(r['layout'], r['scene'], r['seed'], r['arm']): r for r in rows if r['valid']}
    arms = sorted({r['arm'] for r in rows})
    specs = CONTRASTS + [(f'{a}_vs_neutral', 'neutral', a) for a in arms if a.startswith(('patch_', 'sham_'))]
    out = []
    for name, ref, arm in specs:
        for scene in sorted({r['scene'] for r in rows}):
            pairs = [(a, index[(l, s, seed, arm)]) for (l, s, seed, x), a in index.items()
                     if x == ref and s == scene and (l, s, seed, arm) in index]
            if not pairs: continue
            both = [(a, b) for a, b in pairs if a['crossed'] and b['crossed']]
            heights = [b['crossing_pen_bottom_cm'] - a['crossing_pen_bottom_cm'] for a, b in both]
            out.append({'contrast': name, 'scene': scene, 'pairs': len(pairs), 'both_crossed': len(both),
                        'reference_direct_compared_avoided': sum(bool(a['direct']) and bool(b['avoided']) for a, b in both),
                        'reference_avoided_compared_direct': sum(bool(a['avoided']) and bool(b['direct']) for a, b in both),
                        'crossing_height_difference_cm': _spread(heights),
                        'grasp_discordant': sum(bool(a['grasped']) != bool(b['grasped']) for a, b in pairs),
                        'placement_discordant': sum(bool(a['task_success']) != bool(b['task_success']) for a, b in pairs),
                        'identical_outcomes': sum(all(a[k] == b[k] for k in ('crossing_mode', 'direct', 'avoided', 'task_success', 'predictions')) for a, b in pairs)})
    # Same arm, tower vs no-tower: does the tower itself change the path?
    for arm in arms:
        pairs = [(a, index[(l, 'tower', seed, arm)]) for (l, s, seed, x), a in index.items()
                 if x == arm and s == 'no_tower' and (l, 'tower', seed, arm) in index]
        both = [(a, b) for a, b in pairs if a['crossed'] and b['crossed']]
        if pairs:
            out.append({'contrast': f'{arm}_tower_vs_no_tower', 'scene': 'both', 'pairs': len(pairs), 'both_crossed': len(both),
                        'reference_direct_compared_avoided': sum(bool(a['direct']) and bool(b['avoided']) for a, b in both),
                        'reference_avoided_compared_direct': sum(bool(a['avoided']) and bool(b['direct']) for a, b in both),
                        'crossing_height_difference_cm': _spread([b['crossing_pen_bottom_cm'] - a['crossing_pen_bottom_cm'] for a, b in both]),
                        'grasp_discordant': sum(bool(a['grasped']) != bool(b['grasped']) for a, b in pairs),
                        'placement_discordant': sum(bool(a['task_success']) != bool(b['task_success']) for a, b in pairs)})
    return out


def _fmt_rate(rate):
    return '—' if not rate['n'] else f"{rate['k']}/{rate['n']}"


def _fmt(spread):
    return '—' if not spread else f"{spread['median']:.1f} [{spread['min']:.1f}–{spread['max']:.1f}]"


def markdown(suite, rows, cells, contrasts, status):
    lines = [f'# Obstacle suite report: {suite.name}', '',
             f"Status `{status.get('status')}`, {len(rows)} episodes, {sum(not r['valid'] for r in rows)} failed or cancelled (excluded).",
             'Direct/avoided are counts among episodes that crossed the tower bearing while holding the pen. Heights are the pen\'s lowest point, cm.',
             'This report describes outcomes; it does not establish a mechanism.', '']
    for scope in ['all'] + sorted({r['split'] for r in rows}):
        lines += [f'## Cells: {scope}', '',
                  '| Scene | Arm | Valid/n | Grasped | Crossed | Direct | Avoided | through/over/around/none | Crossing height cm | Tower contact | Toppled | Placed | Inactive |',
                  '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for c in cells:
            if c['scope'] != scope: continue
            m = c['modes']
            lines.append(f"| {c['scene']} | {c['arm']} | {c['valid']}/{c['episodes']} | {_fmt_rate(c['grasped'])} | {_fmt_rate(c['crossed'])} | "
                         f"{_fmt_rate(c['direct_given_crossed'])} | {_fmt_rate(c['avoided_given_crossed'])} | "
                         f"{m['through']}/{m['over']}/{m['around']}/{m['none']} | {_fmt(c['crossing_pen_bottom_cm'])} | "
                         f"{_fmt_rate(c['tower_contact'])} | {_fmt_rate(c['tower_toppled'])} | {_fmt_rate(c['placed'])} | {c['inactive']} |")
        lines.append('')
    lines += ['## Paired contrasts', '', 'Discordant pairs: reference direct while compared avoided, and the reverse. Height difference is compared − reference.', '',
              '| Contrast | Scene | Pairs | Both crossed | Ref direct → avoided | Ref avoided → direct | Crossing height diff cm | Grasp discordant | Placement discordant |',
              '|---|---|---|---|---|---|---|---|---|']
    for c in contrasts:
        lines.append(f"| {c['contrast']} | {c['scene']} | {c['pairs']} | {c['both_crossed']} | {c['reference_direct_compared_avoided']} | "
                     f"{c['reference_avoided_compared_direct']} | {_fmt(c['crossing_height_difference_cm'])} | {c['grasp_discordant']} | {c['placement_discordant']} |")
    failed = [r for r in rows if not r['valid']]
    if failed:
        lines += ['', '## Failed or cancelled episodes', ''] + [f"- `{r['episode']}`: {r['termination']}" for r in failed]
    return '\n'.join(lines) + '\n'


def write_report(suite):
    """Writes derived files beside the suite; the sealed suite directory is not modified."""
    suite = Path(suite).resolve()
    results = json.loads((suite/'results.json').read_text()); status = json.loads((suite/'status.json').read_text())
    rows = episode_rows(results); cells, contrasts = cell_table(rows), paired_contrasts(rows)
    outputs = {'markdown': suite.parent/f'{suite.name}-report.md', 'csv': suite.parent/f'{suite.name}-episodes.csv',
               'json': suite.parent/f'{suite.name}-report.json'}
    outputs['markdown'].write_text(markdown(suite, rows, cells, contrasts, status))
    with outputs['csv'].open('w', newline='') as f:
        writer = csv.DictWriter(f, COLUMNS); writer.writeheader(); writer.writerows(rows)
    outputs['json'].write_text(json.dumps({'suite': str(suite), 'status': status, 'cells': cells, 'contrasts': contrasts}, indent=2, allow_nan=False) + '\n')
    return outputs
