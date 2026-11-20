"""Grid geometry, morphological ops, capsules/blobs, and heightfield smoothing."""

from __future__ import annotations

import math

import numpy as np

from .constants import D, W


def dilate8(mask):
    """8-connected binary dilation."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def dilate4(mask):
    """4-connected binary dilation."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def erode4(mask):
    """4-connected binary erosion."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    return out


def erode8(mask):
    """8-connected binary erosion (each True cell has all 8 neighbors True)."""
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
    out[1:, 1:] &= mask[:-1, :-1]
    out[1:, :-1] &= mask[:-1, 1:]
    out[:-1, 1:] &= mask[1:, :-1]
    out[:-1, :-1] &= mask[1:, 1:]
    out[0, :] = False
    out[-1, :] = False
    out[:, 0] = False
    out[:, -1] = False
    return out


def dilate_n(mask, steps, mode=8):
    """Apply `dilate8` or `dilate4` `steps` times."""
    out = mask.copy()
    fn = dilate8 if mode == 8 else dilate4
    for _ in range(steps):
        out = fn(out)
    return out


def paint_capsule(cc, rr, p0, p1, r0, r1=None, warp=None):
    """Smooth tapered capsule mask between p0=(c,r) and p1=(c,r)."""
    if r1 is None:
        r1 = r0
    c0, r_0 = p0
    c1, r_1 = p1
    dx, dz = c1 - c0, r_1 - r_0
    l2 = dx * dx + dz * dz
    if l2 < 1e-6:
        t = np.zeros_like(cc, dtype=np.float64)
    else:
        t = np.clip(((cc - c0) * dx + (rr - r_0) * dz) / l2, 0.0, 1.0)
    px_c = c0 + t * dx
    px_r = r_0 + t * dz
    rad = r0 + t * (r1 - r0)
    if warp is not None:
        rad = rad + warp
    return ((cc - px_c) ** 2 + (rr - px_r) ** 2) <= rad ** 2


def paint_blob(cc, rr, cx, cz, rx, rz, ang_deg=0.0, warp=None):
    """Rotated ellipse mask with optional smooth boundary warp."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    dc = cc - cx
    dr = rr - cz
    u = dc * ca + dr * sa
    v = -dc * sa + dr * ca
    scale = 1.0 if warp is None else (1.0 + warp)
    return (u / (rx * scale)) ** 2 + (v / (rz * scale)) ** 2 <= 1.0


def smooth_noise_field(cc, rr):
    """Smooth deterministic organic perturbation in [-1, 1] without abrupt spikes."""
    return (
        0.50 * np.sin(cc * 0.14 + rr * 0.09)
        + 0.30 * np.cos(cc * 0.08 - rr * 0.17 + 1.2)
        + 0.20 * np.sin((cc + rr) * 0.22 + 0.5)
    )


def ensure_smooth_walkable_heightfield(V, walkable_mask, locked_mask, max_step=18):
    """Relax unlocked walkable vertices until every adjacent walkable pair has |dV| <= max_step."""
    for _ in range(40):
        changed = False
        for r, c in zip(*np.nonzero(walkable_mask & ~locked_mask)):
            vals = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr <= D and 0 <= nc <= W and walkable_mask[nr, nc]:
                    vals.append(int(V[nr, nc]))
            if not vals:
                continue
            lo = max(v - max_step for v in vals)
            hi = min(v + max_step for v in vals)
            cur = int(V[r, c])
            if lo <= hi:
                nv = min(max(cur, lo), hi)
            else:
                nv = int(round(sum(vals) / len(vals)))
            if nv != cur:
                V[r, c] = nv
                changed = True
        if not changed:
            break
