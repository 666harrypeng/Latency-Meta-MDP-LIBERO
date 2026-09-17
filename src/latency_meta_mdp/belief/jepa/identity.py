"""Keep task-specific Belief artifacts distinct from legacy moving-ball levels."""


def data_domain(value):
    level = value.level
    task = getattr(value, "task_id", None)
    action = getattr(value, "action_contract_id", None)
    if task is None and action is None and type(level) is int and level in (1, 2, 3):
        return ("moving_ball", level)
    if level is None and task == "conveyor_sort" and action == "panda_osc_pose_delta_conveyor_v2":
        return (task, action)
    raise ValueError("invalid JEPA task/level/controller identity")
