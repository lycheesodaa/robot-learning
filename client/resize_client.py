"""LeRobot 0.6.1 robot_client that shrinks camera frames to the policy's input size before sending.

Drop-in replacement for `python -m lerobot.async_inference.robot_client`: takes the same
arguments, plus
    --image_size=224x224   HxW to send every camera frame at (default: MolmoAct2's 224x224)

The policy server resizes every frame to the policy's input shape anyway
(helpers.prepare_raw_observation). Doing it here with the same function gives the policy the
same pixels, and the server's own resize becomes a no-op, but each observation shrinks from
2 x 640x480x3 = 1.8 MB to 2 x 224x224x3 = 0.3 MB on the wire.
"""

import sys

import torch

from lerobot.async_inference import robot_client
from lerobot.async_inference.helpers import resize_robot_observation_image
from lerobot.utils.import_utils import register_third_party_plugins

DEFAULT_IMAGE_SIZE = (224, 224)


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


def pop_image_size() -> tuple[int, int]:
    """Take --image_size=HxW out of sys.argv before draccus parses the robot_client config."""
    size = DEFAULT_IMAGE_SIZE
    rest = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith("--image_size="):
            h, w = arg.split("=", 1)[1].lower().split("x")
            size = (int(h), int(w))
        else:
            rest.append(arg)
    sys.argv = rest
    return size


if __name__ == "__main__":
    install(*pop_image_size())
    register_third_party_plugins()
    robot_client.async_client()
