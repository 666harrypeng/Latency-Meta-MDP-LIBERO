# Third-Party Resource Notices

## LIBERO tabletop agentview camera

- Source: `Lifelong-Robot-Learning/LIBERO`
- Revision: `8f1084e3132a39270c3a13ebe37270a43ece2a01`
- Source path: `libero/libero/envs/problems/libero_tabletop_manipulation.py`
- License: MIT
- Local resource: `assets/camera/libero_tabletop_agentview_v1.json`
- Modification: the source quaternion was normalized to unit length without changing its
  orientation; the inherited RoboSuite 45-degree field of view is recorded explicitly.

Every future resource must be listed in `assets/resource_manifest.json` with its immutable source
revision, source and local paths, SHA-256, upstream origin, license, attribution, and modifications
before it is used by the simulator or redistributed.
