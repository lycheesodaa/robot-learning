"""Prepared example manifests: matched pairs, split checks and input images."""
import json
from pathlib import Path


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
