from him_adapter.policy import HoldPolicy


def test_policy_holds_joint_position() -> None:
    policy = HoldPolicy()
    observation = {"signals": {"robot.joint_position": [0.0, 0.1, -0.1]}}
    action = policy.infer(observation)
    assert action["signals"]["robot.desired_joint_position"] == [0.0, 0.1, -0.1]
    assert action["valid_for_ms"] > 0
