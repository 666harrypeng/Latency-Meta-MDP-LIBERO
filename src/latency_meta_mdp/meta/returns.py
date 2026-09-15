"""Short off-policy traces over real, variable-duration decision transitions."""

import torch


def model_visual_batch(visual, indices, device):
    if isinstance(visual, torch.Tensor) and visual.device.type == "cpu" and device.type == "cuda":
        # Gather directly into a small pinned batch, not a second copy of the full bank.
        batch = torch.empty((len(indices), *visual.shape[1:]), dtype=visual.dtype, pin_memory=True)
        torch.index_select(visual, 0, indices.cpu(), out=batch)
        return batch.to(device, non_blocking=True)
    return visual[indices].to(device)


def sequence_indices(next_transition, start_indices, max_steps):
    if type(max_steps) is not int or max_steps < 1:
        raise ValueError("sequence length must be a positive integer")
    positions = torch.arange(len(next_transition), device=next_transition.device)
    if (
        ((next_transition != -1) & (next_transition != positions + 1)).any()
        or (next_transition >= len(next_transition)).any()
        or (start_indices < 0).any()
        or (start_indices >= len(next_transition)).any()
    ):
        raise ValueError("invalid forward episode sequence")
    columns, current = [], start_indices.long()
    for _ in range(max_steps):
        columns.append(current)
        current = torch.where(current >= 0, next_transition[current.clamp_min(0)], -1)
    ids = torch.stack(columns, dim=1)
    return ids, ids >= 0


def trace_targets(reward, discount, next_value, continue_trace, valid_mask):
    if (
        reward.ndim != 2
        or reward.shape[1] < 1
        or any(x.shape != reward.shape for x in (discount, next_value, continue_trace, valid_mask))
        or not valid_mask[:, 0].all()
        or (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any()
    ):
        raise ValueError("invalid padded return sequence")
    r = reward.masked_fill(~valid_mask, 0)
    d = discount.masked_fill(~valid_mask, 0)
    v = next_value.masked_fill(~valid_mask | (d == 0), 0)
    c = continue_trace.masked_fill(~valid_mask, 0)
    if (
        not all(torch.isfinite(x).all() for x in (r, d, v, c))
        or (d < 0).any()
        or (d > 1).any()
        or (c < 0).any()
        or (c > 1).any()
    ):
        raise ValueError("invalid trace return values")
    result = torch.zeros_like(r[:, 0])
    for t in range(r.shape[1] - 1, -1, -1):
        continuation = (
            c[:, t] * valid_mask[:, t + 1] if t + 1 < r.shape[1] else torch.zeros_like(result)
        )
        result = r[:, t] + d[:, t] * ((1 - continuation) * v[:, t] + continuation * result)
    return result.clamp(max=1.0)


def greedy_trace_predictions(model, target, visual, vector, records, admissible, indices, cfg):
    """Double-Q endpoints with Watkins-style cuts at the NEXT recorded action.

    Only selected start states receive gradients. Old behavior propensities are not
    required for this deterministic greedy-target trace. This is not a convergence claim.
    """
    device = next(model.parameters()).device
    ids, valid = sequence_indices(records["next_transition"], indices, cfg["max_trace_steps"])
    safe = ids.clamp_min(0)
    nxt = records["next"][safe]
    unique, inverse = torch.unique(nxt.flatten(), return_inverse=True)
    amp = device.type == "cuda" and cfg.get("q_precision") != "float32"
    values, choices = [], []
    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        for batch in unique.split(128):
            x, v = model_visual_batch(visual, batch, device), vector[batch].to(device)
            legal = admissible[batch].to(device)
            online = model(x, v).float().masked_fill(~legal, -torch.inf)
            choice = online.argmax(1)
            values.append(target(x, v).float().gather(1, choice[:, None])[:, 0])
            choices.append(choice)
        inverse = inverse.to(device)
        next_v = torch.cat(values)[inverse].reshape(ids.shape)
        next_greedy = torch.cat(choices)[inverse].reshape(ids.shape)
        r = records["task_reward"][safe].float().to(device)
        r = r - cfg["cost_multiplier"] * records["cost"][safe].float().to(device) / cfg["budget"]
        d = records["discount"][safe].float().to(device)
        mask = valid.to(device)
        legal_next = admissible[nxt].any(-1).to(device)
        if (mask & (d > 0) & ~legal_next).any():
            raise ValueError("nonterminal sequence lacks a legal successor")
        actions = records["action"][safe].long().to(device)
        c = torch.zeros_like(r)
        c[:, :-1] = (actions[:, 1:] == next_greedy[:, :-1]) * cfg["trace_decay"]
        labels = trace_targets(r, d, next_v, c, mask)
        reach = ((c[:, :-1] > 0) & mask[:, 1:] & (d[:, :-1] > 0)).long().cumprod(1)
        model.last_trace_metrics = {
            "mean_trace_items": float(mask.sum(1).float().mean()),
            "mean_effective_trace_items": float(1 + reach.sum(1).float().mean()),
            "trace_one_step_fraction": float((reach.sum(1) == 0).float().mean()),
        }
    current = records["state"][indices]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        q = model(model_visual_batch(visual, current, device), vector[current].to(device)).float()
        predicted = q.gather(1, records["action"][indices].long().to(device)[:, None])[:, 0]
    return predicted, labels, q
