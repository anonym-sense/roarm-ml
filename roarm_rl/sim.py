"""PyBullet simulation wrapper for the Waveshare RoArm-M2.

This is the single source of truth for joint order, limits and kinematics
used everywhere else in the project (GUI, hardware mirror, and later the RL
environment). Joint order is always:

    0: base     (base_link_to_link1)   -3.1416 .. 3.1416 rad
    1: shoulder  (link1_to_link2)       -1.5708 .. 1.5708 rad
    2: elbow     (link2_to_link3)       -1.0    .. 2.95   rad
    3: gripper   (link3_to_gripper_link) 0.0    .. 1.5    rad

This matches the joint order used by the official `roarm_sdk` package
(`joints_radian_ctrl`), so a radian list produced here can be sent to the
real arm unmodified via roarm_rl.hardware.RoArmHardware.
"""

import os
import tempfile

import pybullet as p
import pybullet_data

JOINT_NAMES = ["base", "shoulder", "elbow", "gripper"]

# roarm_sdk (utils.py calibration_parameters, roarm_m2 "radians_min/max")
# rejects any joint command outside these bounds *before* sending it. The
# official URDF's elbow lower limit (-1.0) is looser than this, so without
# intersecting the two, IK can produce elbow angles the real arm's SDK
# refuses -- sim moves, hardware silently doesn't (the SDK raises, we catch
# it and skip that frame). Intersecting guarantees anything valid in sim is
# always valid to send.
_FIRMWARE_LIMITS_M2 = [(-3.3, 3.3), (-1.9, 1.9), (-0.2, 3.3), (-0.2, 1.9)]

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_URDF_TEMPLATE = os.path.join(_ASSETS_DIR, "urdf", "roarm_m2.urdf.template")
_MESH_DIR = os.path.join(_ASSETS_DIR, "meshes")


def _materialize_urdf():
    """Substitute the mesh directory placeholder and write a temp URDF.

    pybullet's URDF parser wants plain file paths (no ROS $(find ...) or
    xacro), so the template is resolved to an absolute path at load time
    instead of baking a machine-specific path into the repo.
    """
    with open(_URDF_TEMPLATE, "r") as f:
        text = f.read()
    mesh_dir_uri = _MESH_DIR.replace(os.sep, "/")
    text = text.replace("{{MESH_DIR}}", mesh_dir_uri)
    fd, path = tempfile.mkstemp(suffix=".urdf", prefix="roarm_m2_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


class RoArmSim:
    """Thin, explicit wrapper around a PyBullet RoArm-M2 instance."""

    def __init__(self, gui=True, timestep=1.0 / 240.0):
        self.gui = gui
        self.timestep = timestep
        self._client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(timestep, physicsClientId=self._client)

        self._plane = p.loadURDF("plane.urdf", physicsClientId=self._client)

        if gui:
            # We implement our own click-and-drag-the-hand picking (app.py);
            # PyBullet's built-in object dragging would otherwise fight our
            # position-controlled motors for the same links.
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0, physicsClientId=self._client)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=self._client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._client)
            p.changeVisualShape(self._plane, -1, rgbaColor=[0.93, 0.94, 0.96, 1.0],
                                physicsClientId=self._client)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.65,
                cameraYaw=50,
                cameraPitch=-30,
                cameraTargetPosition=[0.12, 0, 0.18],
                physicsClientId=self._client,
            )

        urdf_path = _materialize_urdf()
        try:
            self.robot = p.loadURDF(
                urdf_path,
                basePosition=[0, 0, 0],
                useFixedBase=True,
                flags=p.URDF_USE_SELF_COLLISION,
                physicsClientId=self._client,
            )
        finally:
            os.remove(urdf_path)

        self._joint_indices = []
        self._joint_limits = []
        name_to_index = {}
        for j in range(p.getNumJoints(self.robot, physicsClientId=self._client)):
            info = p.getJointInfo(self.robot, j, physicsClientId=self._client)
            name_to_index[info[1].decode("utf-8")] = j

        ordered_joint_names = [
            "base_link_to_link1",
            "link1_to_link2",
            "link2_to_link3",
            "link3_to_gripper_link",
        ]
        for jn, fw_limit in zip(ordered_joint_names, _FIRMWARE_LIMITS_M2):
            j = name_to_index[jn]
            info = p.getJointInfo(self.robot, j, physicsClientId=self._client)
            self._joint_indices.append(j)
            urdf_lower, urdf_upper = info[8], info[9]
            fw_lower, fw_upper = fw_limit
            self._joint_limits.append((max(urdf_lower, fw_lower), min(urdf_upper, fw_upper)))

        self._ee_link_index = name_to_index["link3_to_hand_tcp"]

        if gui:
            self._style_robot()

        # Matches roarm_sdk's move_init() for roarm_m2, so sim and real-arm
        # "home" line up.
        self.home_radians = [0.0, 0.0, 1.5708, 0.0]
        self.set_joint_targets(self.home_radians, instant=True)

    def _style_robot(self):
        c = self._client
        p.changeVisualShape(self.robot, -1, rgbaColor=[0.22, 0.24, 0.28, 1.0], physicsClientId=c)
        for j in range(p.getNumJoints(self.robot, physicsClientId=c)):
            name = p.getJointInfo(self.robot, j, physicsClientId=c)[12].decode()
            if name == "gripper_link":
                color = [0.98, 0.58, 0.18, 1.0]
            elif name == "hand_tcp":
                continue
            else:
                color = [0.86, 0.88, 0.92, 1.0]
            p.changeVisualShape(self.robot, j, rgbaColor=color, physicsClientId=c)

    def add_target_marker(self, radius=0.012):
        """Visual-only sphere that shows where the IK target currently is."""
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0.2, 0.8, 1.0, 0.85],
                                  physicsClientId=self._client)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis, basePosition=[0, 0, 0],
                                 physicsClientId=self._client)

    @property
    def client_id(self):
        return self._client

    @property
    def joint_limits(self):
        """List of (lower, upper) radian limits, in JOINT_NAMES order."""
        return list(self._joint_limits)

    def clamp(self, radians):
        out = []
        for r, (lo, hi) in zip(radians, self._joint_limits):
            out.append(min(max(r, lo), hi))
        return out

    def set_joint_targets(self, radians, instant=False, max_force=15.0, max_velocity=2.0):
        """Drive the 4 joints towards `radians` (base, shoulder, elbow, gripper).

        `instant=True` teleports the joints (used for reset/home); otherwise
        a position-controlled motor command is issued and the caller is
        expected to keep calling step().
        """
        radians = self.clamp(radians)
        for idx, target in zip(self._joint_indices, radians):
            if instant:
                p.resetJointState(self.robot, idx, target, physicsClientId=self._client)
            else:
                p.setJointMotorControl2(
                    self.robot,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=max_force,
                    maxVelocity=max_velocity,
                    physicsClientId=self._client,
                )
        return radians

    def get_joint_states(self):
        """Current (position, velocity) radians for each of the 4 joints."""
        states = p.getJointStates(self.robot, self._joint_indices, physicsClientId=self._client)
        return [(s[0], s[1]) for s in states]

    def get_ee_pose(self):
        """(xyz, quaternion) of the hand_tcp frame in world coordinates."""
        state = p.getLinkState(self.robot, self._ee_link_index, physicsClientId=self._client)
        return state[4], state[5]

    def solve_ik(self, target_xyz, target_orientation=None):
        """Inverse kinematics for the hand_tcp frame -> 4 joint radians."""
        lower = [lo for lo, _ in self._joint_limits]
        upper = [hi for _, hi in self._joint_limits]
        ranges = [hi - lo for lo, hi in self._joint_limits]
        rest = self.get_joint_states()
        rest_poses = [s[0] for s in rest]

        kwargs = dict(
            bodyUniqueId=self.robot,
            endEffectorLinkIndex=self._ee_link_index,
            targetPosition=target_xyz,
            lowerLimits=lower,
            upperLimits=upper,
            jointRanges=ranges,
            restPoses=rest_poses,
            physicsClientId=self._client,
        )
        if target_orientation is not None:
            kwargs["targetOrientation"] = target_orientation
        solution = p.calculateInverseKinematics(
            maxNumIterations=200, residualThreshold=1e-5, **kwargs
        )
        return self.clamp(list(solution[:4]))

    def step(self):
        p.stepSimulation(physicsClientId=self._client)

    def close(self):
        if self._client is not None:
            try:
                p.disconnect(physicsClientId=self._client)
            except p.error:
                pass  # window was already closed by the user
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
