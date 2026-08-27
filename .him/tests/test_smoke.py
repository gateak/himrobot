import json
from pathlib import Path

from him_adapter.policy import HoldPolicy

CONTRACT = Path(__file__).resolve().parents[1] / "contracts" / "runtime.json"
FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "observation.json"


def test_contract_declares_29_g1_joints() -> None:
    contract = json.loads(CONTRACT.read_text())
    assert contract["observations"][0]["shape"] == [29]
    assert contract["actions"][0]["shape"] == [29]
    assert len(contract["extensions"]["robot"]["joint_order"]) == 29


def test_policy_holds_joint_position() -> None:
    observation = json.loads(FIXTURE.read_text())
    joints = observation["signals"]["robot.joint_position"]
    assert len(joints) == 29
    action = HoldPolicy().infer(observation)
    assert action["signals"]["robot.desired_joint_position"] == joints
    assert action["valid_for_ms"] > 0
