"""Explicit history reader and history-guided forecast fusion for scheduler Q values."""

import torch
from torch import nn

from latency_meta_mdp.meta.information.attention import ForecastBlock, HistoryBlock, VisualTokenizer
from latency_meta_mdp.meta.information.memory_schema import ActionSpec, MemoryBatch


class AttentionStageQ(nn.Module):
    def __init__(self, config, *, normalization=None):
        super().__init__()
        self.config = config
        model, memory = config["model"], config["memory"]
        width, heads, hidden = model["width"], model["num_heads"], model["ffn_width"]
        self.action_spec = ActionSpec(**config["actions"])
        self.time_factor = memory["tick_seconds"] / memory["time_scale"]
        self.visual = VisualTokenizer(width, heads, memory["spatial_tokens"])
        self.proprio = nn.Linear(16, width)
        self.scheduler = nn.ModuleList([nn.Linear(12, width) for _ in range(2)])
        self.action = nn.Linear(7, width)
        self.event = nn.Linear(16, width)
        self.forecast_metadata = nn.Linear(8, width)
        self.modality = nn.Embedding(12, width)
        self.condition = nn.Sequential(nn.Linear(40, width), nn.LayerNorm(width), nn.GELU())
        self.queries = nn.Parameter(torch.randn(model["num_queries"], width) * 0.02)
        self.history = nn.ModuleList([HistoryBlock(width, heads, hidden) for _ in range(2)])
        self.forecast = ForecastBlock(width, heads, hidden)
        self.final_norm = nn.LayerNorm(width)
        self.critical = nn.Sequential(nn.Linear(40, 128), nn.GELU())
        self.readout = nn.Sequential(nn.Linear(model["num_queries"] * width + 128, 512), nn.GELU())
        self.heads = nn.ModuleList([nn.Linear(512, self.action_spec.num_actions) for _ in range(2)])
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        norm = normalization or dict(mean=[0.0] * 16, scale=[1.0] * 16)
        mean, scale = (
            torch.tensor(norm["mean"], dtype=torch.float32),
            torch.tensor(norm["scale"], dtype=torch.float32),
        )
        if (
            mean.shape != (16,)
            or scale.shape != (16,)
            or not torch.isfinite(mean).all()
            or not torch.isfinite(scale).all()
            or not (scale > 0).all()
        ):
            raise ValueError("invalid frozen proprio normalization")
        self.register_buffer("proprio_mean", mean)
        self.register_buffer("proprio_scale", scale)

    def normalized(self, state):
        return (state - self.proprio_mean) / self.proprio_scale

    def _bank(self, t, critical):
        chunks, times, masks = [], [], []
        b = len(critical)

        def add(features, source_ticks, valid, kind):
            features = features.reshape(b, -1, features.shape[-1])
            chunks.append(features + self.modality.weight[kind])
            times.append((source_ticks.reshape(b, -1) - t["tick"][:, None]) * self.time_factor)
            masks.append(valid.reshape(b, -1))

        now, yes = t["tick"][:, None], torch.ones((b, 1), device=critical.device, dtype=torch.bool)
        add(self.proprio(critical[:, :16])[:, None], now, yes, 0)
        for i in range(2):
            add(self.scheduler[i](critical[:, 16 + 12 * i : 28 + 12 * i])[:, None], now, yes, 1 + i)
        add(
            self.action(t["buffer"].masked_fill(~t["buffer_mask"][..., None], 0)),
            t["buffer_ticks"],
            t["buffer_mask"],
            3,
        )
        add(
            self.action(t["execution_action"].masked_fill(~t["execution_mask"][..., None], 0)),
            t["execution_ticks"][..., 0],
            t["execution_mask"],
            4,
        )
        add(
            self.proprio(
                self.normalized(t["execution_proprio"]).masked_fill(
                    ~t["execution_mask"][..., None], 0
                )
            ),
            t["execution_ticks"][..., 1],
            t["execution_mask"],
            5,
        )
        add(
            self.event(t["event_features"].masked_fill(~t["event_mask"][..., None], 0)),
            t["event_ticks"],
            t["event_mask"],
            6,
        )
        add(
            self.proprio(
                self.normalized(t["visual_proprio"]).masked_fill(~t["visual_mask"][..., None], 0)
            ),
            t["visual_ticks"],
            t["visual_mask"],
            7,
        )
        for start, end, dense in ((0, 3, True), (3, t["visual"].shape[1], False)):
            if start == end:
                continue
            valid = t["visual_mask"][:, start:end]
            frames = t["visual"][:, start:end].masked_fill(~valid[..., None, None, None], 0)
            visual = self.visual(frames, source_kind=0, dense=dense)
            shape = visual.shape[:-1]
            source = t["visual_ticks"][:, start:end, None, None].expand(shape)
            mask = valid[..., None, None].expand(shape)
            add(visual, source, mask, 8)
        return torch.cat(chunks, 1), torch.cat(times, 1), torch.cat(masks, 1)

    def forward(self, batch):
        if not isinstance(batch, MemoryBatch):
            raise TypeError("attention Q accepts only public MemoryBatch inputs")
        t = batch.tensors
        if torch.any((t["stage"] != 0) & (t["stage"] != 2)):
            raise ValueError("bound Observe stage is forced, not a Q decision")
        critical = torch.cat((self.normalized(t["critical"][:, :16]), t["critical"][:, 16:]), -1)
        bank, times, mask = self._bank(t, critical)
        query = self.queries[None] + self.condition(critical)[:, None]
        for layer in self.history:
            query = layer(query, bank, times, mask)
        valid = t["forecast_mask"]
        visual = self.visual(
            t["forecast"].masked_fill(~valid[:, None, None, None], 0), source_kind=1, dense=True
        ).flatten(1, 2)
        proprio = self.proprio(
            self.normalized(t["forecast_proprio"]).masked_fill(~valid[:, None], 0)
        )[:, None]
        metadata = self.forecast_metadata(t["forecast_metadata"].masked_fill(~valid[:, None], 0))[
            :, None
        ]
        forecast = torch.cat(
            (
                visual + self.modality.weight[11],
                proprio + self.modality.weight[9],
                metadata + self.modality.weight[10],
            ),
            1,
        )
        forecast_times = ((t["forecast_tick"] - t["tick"]) * self.time_factor)[:, None].expand(
            -1, forecast.shape[1]
        )
        query = self.forecast(
            query, forecast, forecast_times, valid[:, None].expand(-1, forecast.shape[1])
        )
        representation = self.readout(
            torch.cat((self.final_norm(query).flatten(1), self.critical(critical)), -1)
        )
        values = torch.stack([head(representation) for head in self.heads], dim=1)
        return values[torch.arange(len(critical), device=critical.device), (t["stage"] == 2).long()]
