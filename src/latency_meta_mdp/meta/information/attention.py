"""Small attention primitives with separate physical-time and padding semantics."""

import torch
from torch import nn
from torch.nn import functional as F


def rotate_time(x, coordinates):
    """Continuous physical coordinates, interleaved rotary pairs, FP32 phases."""
    width = x.shape[-1]
    if width % 2:
        raise ValueError("rotary dimension must be even")
    frequencies = 10000.0 ** (
        -torch.arange(0, width, 2, device=x.device, dtype=torch.float32) / width
    )
    phase = coordinates.float()[..., None] * frequencies
    cosine, sine = phase.cos().to(x.dtype), phase.sin().to(x.dtype)
    pairs = x.reshape(*x.shape[:-1], width // 2, 2)
    a, b = pairs.unbind(-1)
    return torch.stack((a * cosine - b * sine, a * sine + b * cosine), -1).flatten(-2)


class CrossAttention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads or (width // heads) % 2:
            raise ValueError("head width must be even")
        self.heads = heads
        self.query, self.key, self.value = (nn.Linear(width, width) for _ in range(3))
        self.output = nn.Linear(width, width)

    def forward(self, query, bank, times=None, mask=None):
        b, n, width = query.shape

        def shape(x):
            return x.reshape(b, -1, self.heads, width // self.heads).transpose(1, 2)

        q, k, v = shape(self.query(query)), shape(self.key(bank)), shape(self.value(bank))
        if times is not None:
            k = rotate_time(k, times[:, None, :])
            q = rotate_time(q, torch.zeros((b, 1, n), device=q.device))
        valid = None if mask is None else mask[:, None, None, :]
        result = F.scaled_dot_product_attention(q, k, v, attn_mask=valid, dropout_p=0.0)
        result = self.output(result.transpose(1, 2).reshape(b, n, width))
        if mask is not None:
            result = result * mask.any(-1)[:, None, None]
        return result


def feed_forward(width, hidden):
    return nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, width))


class HistoryBlock(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.query_norm, self.bank_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross = CrossAttention(width, heads)
        self.self_norm = nn.LayerNorm(width)
        self.self_attention = CrossAttention(width, heads)
        self.ffn_norm, self.ffn = nn.LayerNorm(width), feed_forward(width, hidden)

    def forward(self, query, bank, times, mask):
        query = query + self.cross(self.query_norm(query), self.bank_norm(bank), times, mask)
        norm = self.self_norm(query)
        query = query + self.self_attention(norm, norm)
        return query + self.ffn(self.ffn_norm(query))


class ForecastBlock(nn.Module):
    def __init__(self, width, heads, hidden):
        super().__init__()
        self.query_norm, self.bank_norm = nn.LayerNorm(width), nn.LayerNorm(width)
        self.cross = CrossAttention(width, heads)
        self.gate = nn.Linear(2 * width + 1, width)
        self.ffn_norm, self.ffn = nn.LayerNorm(width), feed_forward(width, hidden)

    def forward(self, query, bank, times, mask):
        valid = mask.any(-1)
        update = self.cross(self.query_norm(query), self.bank_norm(bank), times, mask)
        gate = self.gate(
            torch.cat((query, update, valid[:, None, None].expand(-1, query.shape[1], 1)), -1)
        ).sigmoid()
        fused = query + gate * update
        fused = fused + self.ffn(self.ffn_norm(fused))
        return torch.where(valid[:, None, None], fused, query)


class VisualTokenizer(nn.Module):
    def __init__(self, width, heads, spatial_tokens):
        super().__init__()
        self.channel = nn.Sequential(nn.LayerNorm(384), nn.Linear(384, width))
        self.row, self.column = (
            nn.Parameter(torch.randn(14, width) * 0.02),
            nn.Parameter(torch.randn(14, width) * 0.02),
        )
        self.view, self.source = (
            nn.Parameter(torch.randn(2, width) * 0.02),
            nn.Parameter(torch.randn(2, width) * 0.02),
        )
        self.pool_queries = nn.Parameter(torch.randn(spatial_tokens, width) * 0.02)
        self.pool_norm = nn.LayerNorm(width)
        self.pool = CrossAttention(width, heads)

    def forward(self, frames, *, source_kind, dense):
        shape = frames.shape[:-3]
        tokens = self.channel(frames)
        position = (self.row[:, None, :] + self.column[None, :, :]).reshape(196, -1)
        tokens = tokens + position + self.view[:, None, :] + self.source[source_kind]
        if dense:
            return tokens
        bank = tokens.reshape(-1, 196, tokens.shape[-1])
        query = self.pool_queries[None].expand(len(bank), -1, -1)
        pooled = query + self.pool(query, self.pool_norm(bank))
        return pooled.reshape(*shape, 2, len(self.pool_queries), tokens.shape[-1])
