# roarm-ml

Simulation, control and reinforcement learning for the
[Waveshare RoArm-M2](https://www.waveshare.com/wiki/RoArm-M2-S), a 4-DOF desktop
robot arm.

The idea: one PyBullet model of the arm that you can drive by hand, mirror
onto the real robot, and train an RL policy against, all through the same
joint vector so nothing needs translating between sim and hardware.

## What's in here

| Module | What it does |
| --- | --- |
| `roarm_rl/sim.py` | `RoArmSim`: PyBullet wrapper with joint control, FK/IK and joint limits |
| `roarm_rl/app.py` | Interactive 3D GUI: keyboard control plus click-and-drag on the arm |
| `roarm_rl/picking.py` | Mouse-ray math behind the click-and-drag |
| `roarm_rl/hardware.py` | `RoArmHardware`: mirrors joint targets to a real arm via `roarm_sdk` |
| `roarm_rl/gesture.py` | Keyframe gestures (`dance`, `wave`) on a Catmull-Rom spline |
| `roarm_rl/env.py` | `RoArmReachEnv`: Gymnasium reach task |
| `roarm_rl/train.py` | PPO training and evaluation (Stable-Baselines3) |
| `roarm_rl/main.py` | CLI entry point for the GUI |
| `assets/` | RoArm-M2 URDF and meshes (from Waveshare, see [NOTICE.md](NOTICE.md)) |

Joint order everywhere is `[base, shoulder, elbow, gripper]` in radians. This
matches `roarm_sdk.joints_radian_ctrl`, so a joint vector that is valid in the
simulator can be sent straight to the real arm. Sim joint limits are the
intersection of the URDF limits and the SDK's firmware limits, so the sim
never produces a pose the arm would refuse.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**Windows note:** `pybullet` has no official prebuilt wheel for Windows on
recent Python versions, so `pip install pybullet` tries to compile from source
and fails without the
[MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
Either install those, or `pip install pybullet-arm64` instead: a community
fork that ships a precompiled Windows wheel and is a drop-in replacement
(`import pybullet` still works). This project was developed with the fork on
Python 3.11.

`roarm-sdk` and `pyserial` are only needed to drive a real arm; `gymnasium`
and `stable-baselines3` only for training.

## Interactive GUI

```
python -m roarm_rl.main                              # sim only
python -m roarm_rl.main --hw serial --port COM5      # real arm over USB
python -m roarm_rl.main --hw http --host 192.168.4.1 # real arm over WiFi (AP mode default IP)
```

| Key | Action |
| --- | --- |
| `M` | toggle Joint mode / IK-XYZ mode |
| `LEFT` / `RIGHT` | Joint: base. IK: target X |
| `UP` / `DOWN` | Joint: shoulder. IK: target Y |
| `PAGE UP` / `PAGE DOWN` | Joint: elbow. IK: target Z |
| `,` / `.` | gripper close / open |
| `P` | toggle "Mirror to Real Robot" |
| `H` | send HOME pose |
| `T` | toggle hardware torque |
| `ESC` | quit |

You can also click and drag the arm itself to move the hand in 3D. The drag
happens in the plane facing the camera at the depth you grabbed, so orbit the
camera first to reach a different depth.

Mirroring is off by default even when `--hw` connects. Once on, joint targets
are throttled to about 12 Hz so the servo bus isn't flooded.

## Gestures

```
python -m roarm_rl.gesture dance                   # preview in the simulator
python -m roarm_rl.gesture wave --hardware COM9    # play on the real arm
```

Gestures are lists of `(duration, pose)` keyframes joined by a Catmull-Rom
spline, so velocity stays continuous through each pose. The keyframes use
animation-style timing: anticipation before a big move, overshoot and settle,
and the gripper trailing the wrist. Every pose is checked against a
conservative safe range before anything is sent to the arm.

## Reinforcement learning

`RoArmReachEnv` is a Gymnasium environment: move the hand to a random
reachable target point.

- **Observation (17):** joint positions, joint velocities, hand position,
  target position, target minus hand
- **Action (4):** joint-angle increments in `[-1, 1]`, scaled to 0.05 rad per step
- **Reward:** negative distance to target, a small action penalty, +1 on success
- **Episode:** ends within 2 cm of the target, or after 200 steps

```
python -m roarm_rl.train --timesteps 300000 --out runs/ppo_reach
python -m roarm_rl.train --eval runs/ppo_reach/final.zip
```

Training runs headless with PPO and writes checkpoints to
`<out>/checkpoints/` and the final model to `<out>/final.zip`.

## Status

- The URDF loads and simulates with four revolute joints; joint-space control
  reaches commanded targets.
- IK (`RoArmSim.solve_ik`) converges to sub-millimetre accuracy against the
  `hand_tcp` frame on the targets tried so far.
- **The reach policy does not work yet.** A 1M-step PPO run with the default
  settings reached the target in 1 of 20 evaluation episodes, with a mean
  final distance of 12.7 cm. The environment and training loop run end to end;
  the reward, observation or hyperparameters still need work.
- Not built yet: replaying a trained policy on the real arm.

## Automatic push

This repository pushes to GitHub after every commit through a git hook. To
enable it in a fresh clone:

```
git config core.hooksPath .githooks
```

## License

The code is released under the [MIT License](LICENSE).

The robot model in `assets/` comes from Waveshare and is not covered by that
license, and the optional `roarm-sdk` dependency is AGPL-3.0. Details are in
[NOTICE.md](NOTICE.md). This project is not affiliated with Waveshare.
