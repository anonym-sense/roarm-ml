# Third-party notices

The MIT license in [LICENSE](LICENSE) covers the Python code in `roarm_rl/`
and the other files written for this project. It does **not** cover the
following, which belong to their respective owners.

## Robot model (`assets/`)

`assets/urdf/roarm_m2.urdf.template` and `assets/meshes/*.stl` describe the
Waveshare RoArm-M2 and are derived from Waveshare's ROS 2 workspace:

- Source: https://github.com/waveshareteam/roarm_ws
  (`src/roarm_main/roarm_description`)
- Changes: xacro includes and `$(find ...)` package paths were stripped and
  inlined so the URDF loads outside ROS; mesh paths use a `{{MESH_DIR}}`
  placeholder. Kinematics, limits and inertials are unchanged.

That repository does not declare a license, so these files remain the
property of Waveshare and are included here only so the simulator runs out of
the box. If you redistribute them, check with Waveshare first.

## Optional runtime dependency: `roarm-sdk`

The hardware paths (`--hw serial`, `--hw http`, `gesture --hardware`) import
Waveshare's `roarm-sdk`. Its source repository is licensed under AGPL-3.0,
while the package metadata on PyPI says MIT; this project treats it as
AGPL-3.0, the stricter of the two:
https://github.com/waveshareteam/waveshare_roarm_sdk

It is not bundled with this repository; you install it yourself only if you
want to drive a real arm. If you distribute something that combines this code
with `roarm-sdk`, the AGPL's terms apply to that combination.

## Loaded by the web app

The browser page loads [three.js](https://threejs.org) (MIT) from the
jsDelivr CDN and, when the camera is switched on,
[MediaPipe Tasks Vision](https://ai.google.dev/edge/mediapipe) (Apache-2.0)
from jsDelivr with its hand-landmark model from Google's model storage.
None of these are bundled with this repository.

"RoArm" and "Waveshare" are names of Waveshare Electronics. This project is
not affiliated with or endorsed by Waveshare.
