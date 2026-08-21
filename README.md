# Latency Meta-MDP — RoboSuite / LIBERO Asset Backend

This repository is an independent simulation backend for studying asynchronous VLA inference
under latency. It uses MuJoCo and RoboSuite with a single-arm Panda/Franka embodiment and builds an
`dynamic_grasp_lift` L0–L3 moving-ball task family.

LIBERO is a resource source, not the research runtime or algorithmic benchmark:

- selected object assets, scene definitions, BDDL files, language, regions, and goal predicates
  may be reused after provenance and license review;
- the authoritative clock, robot control, dynamic-world motion, phase tracking, recorder, latency
  harness, and policy integration remain project-owned;
- no result from this repository should be described as an official LIBERO benchmark result.

## Current status

G0 runtime foundation and G1 synchronous timing are implemented and certified. The pinned Python
3.10 / RoboSuite 1.5.2 / MuJoCo 3.3.3 environment, validated 2 ms / 20 ms contract, project-owned
split-step loop, direct boundary snapshots, and real Panda timing calibration are complete. No
L0–L3 motion certification, dataset, SFT checkpoint, or algorithm result exists yet.

The approved project boundary is:

```text
MuJoCo physics
  -> RoboSuite-native environment and Panda/Franka controller
  -> project-owned multi-rate temporal contract
  -> project-owned Dynamic Grasp Lift L0–L3 task family
  -> optional, audited LIBERO resources
  -> synchronized data and latency harness
  -> VLA baseline, then belief/action/meta-policy research
```

Formal training is gated on clock, control, contact, and synchronized-data certification.

## Planning documents

- [Architecture specification](docs/design/2026-08-21-robosuite-libero-infra-design.md)
- [RoboSuite/MuJoCo timing audit](docs/research/2026-08-21-robosuite-mujoco-timing-audit.md)
- [Program roadmap](docs/implementation/MASTER_ROADMAP.md)
- [Detailed infrastructure implementation plan](docs/superpowers/plans/2026-08-21-robosuite-infra.md)
- [Project tracker](docs/PROJECT_TRACKER.md)

## Repository policy

- The sibling original Meta-MDP repository is a read-only research reference.
- This repository does not import implementation modules from the original repository.
- Third-party source or assets require an immutable source revision, path, checksum, license, and
  attribution record before entering the repository.
- Generated datasets, checkpoints, videos, and simulation artifacts remain outside Git.
- Commits use Conventional Commits. Pushes require explicit authorization.
