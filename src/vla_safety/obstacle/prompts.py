"""Token counts of task instructions as the policy formats them, without loading the model.

Patching aligns donor and recipient position by position, so their instructions
must tokenize to the same length. The normalisation below mirrors the
checkpoint's `_normalize_question_text` for plain sentences (whitespace,
trailing punctuation, sentences joined by "; ", lowercase); it is a preflight
check only, and the runtime shape check in the patch hook remains authoritative.
"""
import json
from pathlib import Path
import re


def normalise(text):
    text = re.sub(r'\s+', ' ', text).strip().rstrip('.!?').strip()
    parts = [p.strip() for p in re.split(r'[.!?]+', text) if p.strip()]
    return '; '.join(parts).lower()


def token_counts(config, root):
    from tokenizers import Tokenizer
    root = Path(root)
    model = json.loads((root/'configs/models.json').read_text())[config['policy']['model']]
    tokenizer = Tokenizer.from_file(str(root/'artifacts/models'/config['policy']['model']/model['revision']/'tokenizer.json'))
    # Counted inside the template's leading clause so boundary merges match.
    return {key: len(tokenizer.encode(f'The task is to {normalise(text)}. The setup is').ids)
            for key, text in config['task']['instructions'].items()}


def check_patch_prompts(config, root, arms=None):
    """Arms whose donor and recipient instructions differ in token count."""
    counts = token_counts(config, root)
    problems = []
    for arm in arms or config['arms']:
        spec = config['arms'][arm]
        if 'donor' in spec and counts[spec['donor']] != counts[spec['instruction']]:
            problems.append(f"{arm}: {spec['instruction']}={counts[spec['instruction']]} tokens, {spec['donor']}={counts[spec['donor']]}")
    return counts, problems
