"""Versioned JSON transport so incompatible simulator stacks need not import torch."""
import base64
from io import BytesIO
from PIL import Image
from vla_safety.types import Observation


def encode_observation(obs):
    images = []
    for image in obs.images:
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buf = BytesIO()
        image.convert('RGB').save(buf, format='PNG')
        images.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    return {'version': 1, 'images_png_base64': images, 'instruction': obs.instruction,
            'state': obs.state.tolist(), 'embodiment': obs.embodiment, 'timestamp': obs.timestamp}


def decode_observation(data):
    if data['version'] != 1:
        raise ValueError('Unsupported wire version')
    images = tuple(Image.open(BytesIO(base64.b64decode(x, validate=True))).convert('RGB')
                   for x in data['images_png_base64'])
    return Observation(images, data['instruction'], data['state'], data['embodiment'], data['timestamp'])


def encode_action(action):
    return {'version': 1, 'values': action.values.tolist(), 'embodiment': action.embodiment,
            'control_mode': action.control_mode, 'frame': action.frame,
            'units': list(action.units), 'normalization': action.normalization}
