"""LeRobot 0.6.1 robot_client that shrinks camera frames to the policy's input size before sending.

Drop-in replacement for `python -m lerobot.async_inference.robot_client`: takes the same
arguments, plus
    --image_size=224x224   HxW to send every camera frame at; empty or 'none' sends frames as
                           captured. Overrides `image_size:` in the config YAML, which in turn
                           overrides the default (MolmoAct2's 224x224)
    --no_home              leave the arm where it is on exit instead of returning it to its
                           start pose (see home_on_exit.py)

The policy server resizes every frame to the policy's input shape anyway
(helpers.prepare_raw_observation). Doing it here with the same function gives the policy the
same pixels, and the server's own resize becomes a no-op, but each observation shrinks from
2 x 640x480x3 = 1.8 MB to 2 x 224x224x3 = 0.3 MB on the wire.
"""

import atexit
import os
import sys
import tempfile

import torch
import yaml

from lerobot.async_inference import robot_client
from lerobot.async_inference.helpers import resize_robot_observation_image
from lerobot.utils.import_utils import register_third_party_plugins

import home_on_exit

DEFAULT_IMAGE_SIZE = (224, 224)
_MISSING = object()


def resize_frame(frame, height: int, width: int):
    """(H, W, C) uint8 array -> (height, width, C), exactly as the server would resize it."""
    if frame.shape[:2] == (height, width):
        return frame
    resized = resize_robot_observation_image(torch.from_numpy(frame), (frame.shape[2], height, width))
    return resized.permute(1, 2, 0).contiguous().numpy()


def install(height: int, width: int) -> None:
    """Patch robot_client so every RobotClient (or subclass) built afterwards sends resized frames."""
    make_robot = robot_client.make_robot_from_config
    map_features = robot_client.map_robot_keys_to_lerobot_features

    def make_resizing_robot(config):
        robot = make_robot(config)
        cam_keys = [k for k, v in robot.observation_features.items() if isinstance(v, tuple)]
        get_observation = robot.get_observation

        def get_resized_observation():
            obs = get_observation()
            for k in cam_keys:
                obs[k] = resize_frame(obs[k], height, width)
            return obs

        robot.get_observation = get_resized_observation
        return robot

    def resized_features(robot):
        # Tell the server the shapes it will actually receive.
        features = map_features(robot)
        for ft in features.values():
            if ft["dtype"] in ("image", "video"):
                ft["shape"] = (height, width, ft["shape"][2])
        return features

    robot_client.make_robot_from_config = make_resizing_robot
    robot_client.map_robot_keys_to_lerobot_features = resized_features


def _parse_size(value) -> tuple[int, int] | None:
    """'224x224' -> (224, 224); None, '' or 'none' -> None (send frames as captured)."""
    if value is None or str(value).strip().lower() in ("", "none", "null", "full"):
        return None
    h, w = str(value).lower().split("x")
    return int(h), int(w)


def _pop_yaml_image_size():
    """Remove image_size from the --config_path YAML (draccus rejects unknown keys) and return it.

    The config is rewritten to a temporary copy without the key; returns _MISSING if absent."""
    for i, arg in enumerate(sys.argv):
        if arg.startswith("--config_path="):
            path = arg.split("=", 1)[1]
            break
    else:
        return _MISSING
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "image_size" not in cfg:
        return _MISSING
    value = cfg.pop("image_size")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    with tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
    atexit.register(os.remove, tmp.name)
    sys.argv[i] = f"--config_path={tmp.name}"
    return value


def pop_image_size(default: tuple[int, int] | None = DEFAULT_IMAGE_SIZE) -> tuple[int, int] | None:
    """HxW to shrink frames to, or None to send them as captured, before draccus parses the config.

    --image_size=HxW on the command line wins, then image_size in the YAML, then `default`.
    Both are taken out of sys.argv / the config so draccus never sees them."""
    yaml_value = _pop_yaml_image_size()
    size = default if yaml_value is _MISSING else _parse_size(yaml_value)
    rest = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--image_size="):
            size = _parse_size(arg.split("=", 1)[1])
        else:
            rest.append(arg)
    sys.argv = rest
    print(f"[resize] sending frames at {f'{size[0]}x{size[1]} (HxW)' if size else 'capture resolution'}")
    return size


if __name__ == "__main__":
    if size := pop_image_size():
        install(*size)
    if not home_on_exit.pop_no_home():
        home_on_exit.install()
    register_third_party_plugins()
    robot_client.async_client()
