"""Run LeRobot's async-inference PolicyServer with MolmoAct2 enabled.

lerobot 0.6.1's policy server only accepts the policy types listed in
`lerobot.async_inference.constants.SUPPORTED_POLICIES`, which does not include
"molmoact2". This launcher adds it at runtime and then starts the normal server,
so all of the server's CLI flags still apply:

    python serve_molmoact2.py --host=127.0.0.1 --port=8080 --fps=30
"""

from lerobot.async_inference import policy_server
from lerobot.utils.import_utils import register_third_party_plugins

if "molmoact2" not in policy_server.SUPPORTED_POLICIES:
    policy_server.SUPPORTED_POLICIES.append("molmoact2")

if __name__ == "__main__":
    register_third_party_plugins()
    policy_server.serve()
