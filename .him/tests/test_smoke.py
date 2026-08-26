from him_adapter.policy import HoldPolicy


def test_policy_holds_joint_position() -> None:
    # 22 = Booster K1 DOF (see .him/contracts/runtime.json extensions.robot).
    joints = [0.0] * 22
    joints[2] = 0.1
    joints[6] = -0.1
    policy = HoldPolicy()
    observation = {"signals": {"robot.joint_position": joints}}
    action = policy.infer(observation)
    assert action["signals"]["robot.desired_joint_position"] == joints
    assert action["valid_for_ms"] > 0
