import os

# Must be set before RoboSuite imports its rendering bindings during test collection.
os.environ.setdefault("MUJOCO_GL", "egl")
