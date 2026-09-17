"""Contact-gated conveyor traction and multi-object lifecycle on the physics clock."""

import numpy as np

from latency_meta_mdp.envs.conveyor.outcomes import GoalDeliveryTracker, ParcelLedger


class ConveyorWorld:
    def __init__(self, env, spec, arrivals):
        self.env, self.spec, self.arrivals = env, spec, arrivals
        self.ledger = ParcelLedger(supply_ticks=spec.supply_ticks, drain_ticks=spec.drain_ticks)
        self.goal = GoalDeliveryTracker(
            center=spec.goal_center,
            radius=spec.goal_radius_m,
            ball_radius=spec.scene.ball_radius_m,
            resting_height=spec.belt_top_z + spec.scene.ball_radius_m,
            min_lift=spec.minimum_lift_m,
            grasp_ticks=spec.grasp_ticks,
            release_width_m=spec.release_width_m,
            release_opening_delta_m=spec.release_opening_delta_m,
        )
        self.next_arrival = 0
        self.drive_invocations = 0
        self.collision_events, self.touching_pairs = [], set()
        self.last_gripper_command = -1.0

    def position(self, parcel_id):
        # A free joint's translation is current even before step1 refreshes xpos.
        return np.array(
            self.env.sim.data.get_joint_qpos(self.env.parcels[parcel_id].joints[0])[:3], copy=True
        )

    def before_step(self, env, time_us):
        if time_us % 20000 == 0 and self.next_arrival < len(self.arrivals):
            event = self.arrivals[self.next_arrival]
            tick = time_us // 20000
            if event.tick < tick:
                raise RuntimeError("missed scheduled birth; do not silently resample")
            if event.tick == tick:
                radius = self.spec.scene.ball_radius_m
                position = np.array([*event.xy, self.spec.belt_top_z + radius + 0.0015])
                for parcel_id, status in self.ledger.statuses.items():
                    if (
                        status == "active"
                        and np.linalg.norm(self.position(parcel_id) - position) < 2 * radius
                    ):
                        raise RuntimeError(
                            "scheduled birth overlaps active parcel; scene protocol invalid"
                        )
                env.activate(event, position)
                self.ledger.spawn(event.parcel_id, tick)
                self.next_arrival += 1
        return {}

    def on_control(self, sample):
        self.last_gripper_command = float(sample.action[-1])

    def after_step1(self, point):
        env, spec = self.env, self.spec
        data, model = env.sim.data, env.sim.model
        data.xfrc_applied[env.parcel_body_ids] = 0
        supported, touching = set(), set()
        for contact in data.contact[: data.ncon]:
            g1, g2 = int(contact.geom1), int(contact.geom2)
            a, b = env.contact_to_parcel.get(g1), env.contact_to_parcel.get(g2)
            if env.belt_geom_id in (g1, g2) and abs(contact.frame[2]) > 0.7:
                parcel_id = b if g1 == env.belt_geom_id else a
                if parcel_id is not None:
                    supported.add(parcel_id)
            if a is not None and b is not None and a != b:
                touching.add(tuple(sorted((a, b))))
        for pair in sorted(touching - self.touching_pairs):
            self.collision_events.append({"time_us": point.time_us, "parcels": pair})
        self.touching_pairs = touching
        changed = False
        if point.at_formal_boundary:
            env.boundary_reward = 0.0
            tick = point.formal_tick_index
            for parcel_id, status in list(self.ledger.statuses.items()):
                if status != "active":
                    continue
                position = self.position(parcel_id)
                grasped = env._check_grasp(env.robots[0].gripper, env.parcels[parcel_id])
                robot = env.robots[0]
                fingers = data.qpos[robot._ref_gripper_joint_pos_indexes[robot.arms[0]]]
                success = self.goal.observe(
                    parcel_id,
                    tick,
                    position,
                    grasped=grasped,
                    opening=self.last_gripper_command < 0,
                    gripper_width=float(fingers[0] - fingers[1]),
                )
                if success or position[2] < spec.miss_height_z:
                    self.ledger.finish(parcel_id, "success" if success else "miss", tick)
                    env.boundary_reward += float(success)
                    env.park(parcel_id)
                    changed = True
            env.done = self.ledger.advance(tick)
            if env.done:
                for parcel_id, status in self.ledger.statuses.items():
                    if status == "timeout":
                        env.park(parcel_id)
                        changed = True
        if changed:
            env.sim.forward()
        for parcel_id in supported:
            if self.ledger.statuses.get(parcel_id) != "active":
                continue
            p = self.position(parcel_id)
            if np.any(
                np.abs(p[:2] - spec.transport_center_xy) > np.array(spec.transport_size_xy) / 2
            ):
                continue
            body = env.parcel_body_ids[parcel_id]
            velocity = data.get_joint_qvel(env.parcels[parcel_id].joints[0])[1]
            acceleration = np.clip(
                (spec.belt_speed_mps - velocity) / spec.drive_time_constant_s,
                -spec.max_drive_accel_mps2,
                spec.max_drive_accel_mps2,
            )
            # COM traction is an explicit approximation, not a moving-surface solver.
            data.xfrc_applied[body, 1] = model.body_mass[body] * acceleration
            self.drive_invocations += 1

    def trace(self, tick):
        return {
            "tick": tick,
            "parcels": [
                {
                    "parcel_id": i,
                    "position": self.position(i).tolist(),
                    "velocity": np.array(
                        self.env.sim.data.get_joint_qvel(self.env.parcels[i].joints[0])
                    ).tolist(),
                }
                for i, status in self.ledger.statuses.items()
                if status == "active"
            ],
        }
