"""Gymnasium reach task for the RoArm-M2.

Goal: drive the hand_tcp to a random reachable target point. Actions are
joint-angle increments in the same [base, shoulder, elbow, gripper] order as
roarm_rl.sim, so a trained policy's joint vector can be replayed on hardware
through RoArmHardware.set_joint_targets without any translation.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pybullet as p

from roarm_rl.sim import RoArmSim

ACTION_SCALE = 0.05  # rad per step at full action
SUBSTEPS = 8  # physics steps per control step
MAX_STEPS = 200
SUCCESS_DIST = 0.02  # m


class RoArmReachEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, seed=None):
        super().__init__()
        self.sim = RoArmSim(gui=False)
        self._limits = np.array(self.sim.joint_limits, dtype=np.float64)
        self._lo, self._hi = self._limits[:, 0], self._limits[:, 1]

        self.action_space = spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

        self._rng = np.random.default_rng(seed)
        self._q = None
        self._target = None
        self._steps = 0

    def _random_joints(self):
        return self._rng.uniform(self._lo, self._hi)

    def _ee(self):
        return np.array(self.sim.get_ee_pose()[0])

    def _obs(self):
        states = self.sim.get_joint_states()
        pos = np.array([s[0] for s in states])
        vel = np.array([s[1] for s in states])
        ee = self._ee()
        return np.concatenate([pos, vel / 10.0, ee, self._target, self._target - ee]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        start = self._random_joints()
        goal_q = self._random_joints()
        self.sim.set_joint_targets(list(goal_q), instant=True)
        self._target = self._ee()

        self.sim.set_joint_targets(list(start), instant=True)
        self._q = np.array(start, dtype=np.float64)
        self._steps = 0
        for _ in range(SUBSTEPS):
            self.sim.step()
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._q = np.clip(self._q + action * ACTION_SCALE, self._lo, self._hi)
        self.sim.set_joint_targets(list(self._q))
        for _ in range(SUBSTEPS):
            self.sim.step()
        self._steps += 1

        dist = float(np.linalg.norm(self._ee() - self._target))
        reward = -dist - 0.01 * float(np.sum(action ** 2))
        success = dist < SUCCESS_DIST
        if success:
            reward += 1.0
        terminated = bool(success)
        truncated = self._steps >= MAX_STEPS
        return self._obs(), reward, terminated, truncated, {"dist": dist, "success": success}

    def close(self):
        self.sim.close()
