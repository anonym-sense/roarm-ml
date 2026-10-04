"""Thin wrapper around the official `roarm_sdk` package for mirroring
simulated joint targets onto a real Waveshare RoArm-M2.

Verified against waveshareteam/waveshare_roarm_sdk source (generate.py /
roarm.py):
  - roarm(roarm_type="roarm_m2", port=..., baudrate=...) for serial,
    or roarm(roarm_type="roarm_m2", host=...) for WiFi/HTTP (control-only,
    no feedback).
  - joints_radian_ctrl(radians=[base, shoulder, elbow, gripper], speed, acc)
    sends all 4 joints in one command -- same order as roarm_rl.sim.
  - Firmware-level radian safety limits for roarm_m2 are
    [-3.3, -1.9, -0.2, -0.2] .. [3.3, 1.9, 3.3, 1.9], which is wider than
    the URDF limits roarm_rl.sim already clamps to, so anything that is
    valid in the simulator is safe to send to the real arm.
"""

import logging

log = logging.getLogger(__name__)


class RoArmHardwareError(RuntimeError):
    pass


class RoArmHardware:
    """Connects to a real RoArm-M2 and mirrors joint-radian targets to it.

    Not connected by default -- the GUI opts in explicitly so you can run
    the simulator with no arm plugged in at all.
    """

    def __init__(self, speed=400, acc=10):
        self._arm = None
        self.speed = speed
        self.acc = acc
        self.mode = None  # "serial" | "http"

    @property
    def connected(self):
        return self._arm is not None

    def connect_serial(self, port, baudrate=115200):
        self._arm = self._new_arm(port=port, baudrate=baudrate)
        self.mode = "serial"

    def connect_http(self, host):
        self._arm = self._new_arm(host=host)
        self.mode = "http"

    def _new_arm(self, **kwargs):
        try:
            from roarm_sdk.roarm import roarm
        except ImportError as e:
            raise RoArmHardwareError(
                "roarm_sdk is not installed. Run: pip install roarm-sdk"
            ) from e
        try:
            return roarm(roarm_type="roarm_m2", **kwargs)
        except Exception as e:
            raise RoArmHardwareError(f"Could not connect to RoArm-M2: {e}") from e

    def set_joint_targets(self, radians):
        """radians: [base, shoulder, elbow, gripper], already limit-clamped."""
        if not self.connected:
            raise RoArmHardwareError("not connected")
        self._arm.joints_radian_ctrl(radians=list(radians), speed=self.speed, acc=self.acc)

    def get_joint_positions(self):
        """Measured [base, shoulder, elbow, gripper] radians, or None if the read failed.

        Serial only: the arm sends no feedback over HTTP.
        """
        if not self.connected:
            raise RoArmHardwareError("not connected")
        value = self._arm.joints_radian_get()
        return list(value) if isinstance(value, list) and len(value) == 4 else None

    def home(self):
        if not self.connected:
            raise RoArmHardwareError("not connected")
        self._arm.move_init()

    def set_torque(self, enabled):
        if not self.connected:
            raise RoArmHardwareError("not connected")
        self._arm.torque_set(1 if enabled else 0)

    def disconnect(self):
        if self._arm is not None:
            try:
                self._arm.disconnect()
            except Exception:
                log.exception("error disconnecting from RoArm")
            self._arm = None
            self.mode = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()
