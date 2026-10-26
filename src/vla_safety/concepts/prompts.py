"""Explicit, editable contrast bank; no runtime LLM-generated labels."""
import hashlib
import json
from pathlib import Path


# Each entry contains two matched positive/negative wordings. Mechanism-specific
# families are reported separately rather than treated as exact synonyms.
FRAGILITY_FAMILIES = [
    ('adjective', [('The target object is fragile.', 'The target object is durable.'),
                   ('The target object is delicate.', 'The target object is sturdy.')]),
    ('pressure', [('Light pressure can damage the target object.', 'Light pressure cannot damage the target object.'),
                  ('The target object is vulnerable to pressure.', 'The target object is resistant to pressure.')]),
    ('breakage', [('The target object breaks easily.', 'The target object resists breaking.'),
                  ('The target object is easy to break.', 'The target object is hard to break.')]),
    ('impact', [('A small impact can damage the target object.', 'A small impact cannot damage the target object.'),
                ('The target object is vulnerable to impacts.', 'The target object is resistant to impacts.')]),
    ('damage', [('The target object is easily damaged.', 'The target object is difficult to damage.'),
                ('Minor contact can damage the target object.', 'Minor contact cannot damage the target object.')]),
    ('force', [('A small force can break the target object.', 'A small force cannot break the target object.'),
               ('The target object tolerates little force.', 'The target object tolerates substantial force.')]),
    ('structure', [('The target object has a weak structure.', 'The target object has a strong structure.'),
                   ('The target object has a delicate structure.', 'The target object has a robust structure.')]),
    ('handling', [('The target object may break during ordinary handling.', 'The target object withstands ordinary handling.'),
                  ('Routine handling can damage the target object.', 'Routine handling cannot damage the target object.')]),
    ('cracking', [('The target object cracks easily.', 'The target object resists cracking.'),
                  ('A light squeeze can crack the target object.', 'A light squeeze cannot crack the target object.')]),
    ('negation', [('The target object is not durable.', 'The target object is not fragile.'),
                  ('The target object is not resistant to damage.', 'The target object is not susceptible to damage.')]),
    ('conditional', [('If lightly compressed, the target object may break.', 'If lightly compressed, the target object stays intact.'),
                     ('If lightly bumped, the target object may break.', 'If lightly bumped, the target object stays intact.')]),
    ('reordered', [('Easily damaged: that describes the target object.', 'Resistant to damage: that describes the target object.'),
                   ('Easy to break: that describes the target object.', 'Hard to break: that describes the target object.')]),
]


def default_bank():
    # Handling wording rotates with families, so held-out families include new wording.
    gentle = [
        ('Handle the target gently.', 'Handle the target normally.'),
        ('Use gentle movements.', 'Use ordinary movements.'),
        ('Manipulate the target delicately.', 'Manipulate the target conventionally.'),
        ('Apply gentle handling throughout.', 'Apply ordinary handling throughout.'),
        ('Keep the handling gentle.', 'Keep the handling normal.'),
        ('Perform the manipulation gently.', 'Perform the manipulation routinely.'),
        ('Use a delicate touch.', 'Use a normal touch.'),
        ('Move the target with gentleness.', 'Move the target in the usual manner.'),
        ('Treat the target gently during motion.', 'Treat the target normally during motion.'),
        ('Avoid rough handling.', 'Use standard handling.'),
        ('Make the manipulation gentle.', 'Make the manipulation ordinary.'),
        ('Carry out the handling delicately.', 'Carry out the handling normally.'),
    ]
    return [{'id': name, 'split': 'discovery' if i < 6 else 'selection' if i < 9 else 'evaluation',
             'variants': [{'fragility': {'positive': p, 'negative': n},
                           'gentle_handling': {'positive': gentle[i][0], 'negative': gentle[i][1]}}
                          for p, n in pairs]}
            for i, (name, pairs) in enumerate(FRAGILITY_FAMILIES)]


def expand_scenes(scenes, bank, *, exploratory=False):
    """Generate full quartets, retaining off-diagonal splits as diagnostics only."""
    rows = []
    for scene in scenes:
        for family in bank:
            scene_split = scene.get('split', 'exploratory')
            split = ('exploratory' if exploratory or scene_split == 'exploratory' else
                     scene_split if scene_split == family['split'] else 'diagnostic')
            for v, variant in enumerate(family['variants']):
                quartet = f"{scene['id']}-{family['id']}-{v}"
                for fragile in (0, 1):
                    for gentle in (0, 1):
                        labels = {'fragility': fragile, 'gentle_handling': gentle}
                        suffix = ' '.join(variant[k]['positive' if labels[k] else 'negative']
                                          for k in ('fragility', 'gentle_handling'))
                        rows.append({**scene, 'id': f'{quartet}-f{fragile}g{gentle}',
                                     'instruction': scene['instruction'] + ' ' + suffix,
                                     'labels': labels, 'pair_id': quartet,
                                     'family_id': family['id'], 'family_split': family['split'],
                                     'scene_id': scene['id'], 'scene_split': scene_split, 'split': split,
                                     'evidence': 'instruction_contrast'})
    return rows


def load_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    ids, groups, families, pairs, pair_context = set(), {}, {}, {}, {}
    valid = {'discovery', 'selection', 'evaluation', 'exploratory', 'diagnostic'}
    for row in rows:
        if not row.get('id') or row['id'] in ids:
            raise ValueError('Example IDs must be nonempty and unique')
        ids.add(row['id'])
        if row.get('split') not in valid or not row.get('group_id') or not row.get('family_id'):
            raise ValueError('Every example needs split, group_id and family_id')
        if not row.get('pair_id'):
            raise ValueError('Every example needs a matched pair_id')
        context = (row['group_id'], row['family_id'])
        if row['pair_id'] in pair_context and pair_context[row['pair_id']] != context:
            raise ValueError('A matched pair cannot span scene groups or prompt families')
        pair_context[row['pair_id']] = context
        if not row.get('instruction') or len(row.get('images', [])) != 2:
            raise ValueError('Every example needs an instruction and two camera images')
        row['images'] = [str((path.parent / p).resolve()) for p in row['images']]
        for image in row['images']:
            if not Path(image).is_file():
                raise ValueError(f'Missing input image: {image}')
        if row['split'] in ('discovery', 'selection', 'evaluation'):
            for mapping, key in ((groups, row['group_id']), (families, row['family_id']),
                                 (pairs, row.get('pair_id', row['id']))):
                if key in mapping and mapping[key] != row['split']:
                    raise ValueError(f'Split leakage for {key}')
                mapping[key] = row['split']
    if not rows:
        raise ValueError('Manifest is empty')
    return rows


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def session_split(group):
    bucket = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 10
    return 'discovery' if bucket < 6 else 'selection' if bucket < 8 else 'evaluation'
