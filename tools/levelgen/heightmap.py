"""Heightmap tile synthesis and original cliff kit stamping."""

from __future__ import annotations

import numpy as np

from .constants import (
    CLIFF_DIAG,
    CLIFF_ORTH,
    D,
    KIND_CLIFF,
    KIND_CRAMP_L,
    KIND_CRAMP_U,
    PATH_OPEN,
    PATH_SLOPEFORCE,
    PATH_SOLID_ALL,
    W,
)


def make_bilinear_tile(nw, ne, sw, se):
    """Create a 32x32 uint8 heightmap tile with exact corner values at [0,0],[0,31],[31,0],[31,31]."""
    u = np.linspace(0.0, 1.0, 32)[None, :]
    v = np.linspace(0.0, 1.0, 32)[:, None]
    vals = (
        nw * (1.0 - u) * (1.0 - v)
        + ne * u * (1.0 - v)
        + sw * (1.0 - u) * v
        + se * u * v
    )
    arr = np.rint(vals).astype(np.uint8)
    arr[0, 0] = int(nw)
    arr[0, 31] = int(ne)
    arr[31, 0] = int(sw)
    arr[31, 31] = int(se)
    return arr


def stamp_cliff_kit(kind, walk):
    """Stamp original Level1 cliff grammar on CRAMP_L / CRAMP_U / CLIFF without touching walk."""
    hm = np.zeros((D, W), np.uint16)
    assigned = np.zeros((D, W), bool)
    CARD = (("N", -1, 0), ("S", 1, 0), ("W", 0, -1), ("E", 0, 1))
    OPP = {"N": "S", "S": "N", "W": "E", "E": "W"}
    STEP = {n: (dr, dc) for n, dr, dc in CARD}
    LOW_TO_TIP = {
        frozenset(("N", "E")): "SW",
        frozenset(("N", "W")): "SE",
        frozenset(("S", "E")): "NW",
        frozenset(("S", "W")): "NE",
    }
    TIP_CLIFF = {
        "SW": ("W", "S"), "SE": ("E", "S"), "NW": ("N", "W"), "NE": ("N", "E"),
    }

    def inb(r, c):
        return 0 <= r < D and 0 <= c < W

    def lows_at(r, c):
        return [name for name, dr, dc in CARD if inb(r + dr, c + dc) and walk[r + dr, c + dc]]

    classes = {}
    for r, c in zip(*np.nonzero(kind == KIND_CRAMP_L)):
        lows = lows_at(r, c)
        key = frozenset(lows)
        if len(lows) == 2 and key in LOW_TO_TIP:
            classes[(r, c)] = ("diag", LOW_TO_TIP[key])
        elif len(lows) >= 1:
            facing = lows[0]
            for f in lows:
                odr, odc = STEP[OPP[f]]
                nr, nc = r + odr, c + odc
                if inb(nr, nc) and kind[nr, nc] in (KIND_CRAMP_U, KIND_CLIFF):
                    facing = f
                    break
            classes[(r, c)] = ("orth", facing)
        else:
            facing = "S"
            best = 1e18
            for name, dr, dc in CARD:
                for dist in range(1, 6):
                    nr, nc = r + dr * dist, c + dc * dist
                    if inb(nr, nc) and walk[nr, nc]:
                        if dist < best:
                            best, facing = dist, name
                        break
            classes[(r, c)] = ("orth", facing)

    # Pass 1: diagonals (181 on CRAMP_L, 180 on CRAMP_U/CLIFF; never 182 on walkable floor)
    for (r, c), (typ, tip) in classes.items():
        if typ != "diag":
            continue
        fx, fy = CLIFF_DIAG[tip]
        hm[r, c] = 181 | (fx << 15) | (fy << 14)
        assigned[r, c] = True
        cliff_dirs = TIP_CLIFF[tip]
        dr = STEP[cliff_dirs[0]][0] + STEP[cliff_dirs[1]][0]
        dc = STEP[cliff_dirs[0]][1] + STEP[cliff_dirs[1]][1]
        ur, uc = r + dr, c + dc
        if inb(ur, uc) and kind[ur, uc] in (KIND_CRAMP_U, KIND_CLIFF) and not assigned[ur, uc]:
            hm[ur, uc] = 180 | (fx << 15) | (fy << 14)
            assigned[ur, uc] = True

    # Pass 2: orth stacks
    for (r, c), (typ, facing) in classes.items():
        if assigned[r, c] or typ != "orth":
            continue
        lo_tid, up_tid, fx, fy = CLIFF_ORTH[facing]
        hm[r, c] = lo_tid | (fx << 15) | (fy << 14)
        assigned[r, c] = True
        odr, odc = STEP[OPP[facing]]
        ur, uc = r + odr, c + odc
        if inb(ur, uc) and kind[ur, uc] in (KIND_CRAMP_U, KIND_CLIFF) and not assigned[ur, uc]:
            hm[ur, uc] = up_tid | (fx << 15) | (fy << 14)
            assigned[ur, uc] = True

    # Pass 3: remaining CRAMP_U
    for r, c in zip(*np.nonzero((kind == KIND_CRAMP_U) & ~assigned)):
        facing = "S"
        for name, dr, dc in CARD:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and (walk[nr, nc] or kind[nr, nc] == KIND_CRAMP_L):
                facing = name
                break
        _, up_tid, fx, fy = CLIFF_ORTH[facing]
        hm[r, c] = up_tid | (fx << 15) | (fy << 14)
        assigned[r, c] = True

    hm[kind == KIND_CLIFF] = 0
    assigned[kind == KIND_CLIFF] = True
    return hm, assigned


def build_heightmaps(world, orig):
    """Build exact vertex-matched hm_grid and hm_tiles list (<= 300 tiles)."""
    kind = world["kind"]
    walk = world["walk"]
    c00, c01, c10, c11 = world["c00"], world["c01"], world["c10"], world["c11"]

    hm_grid, _ = stamp_cliff_kit(kind, walk)

    tile_bank = list(orig.hm_tiles)  # 0..247

    exact_lut = {
        (255, 255, 255, 255): 0,
        (109, 109, 109, 109): 5,
        (85, 85, 85, 85): 6,
        (67, 67, 67, 67): 7,
        (41, 41, 41, 41): 8,
        (128, 128, 128, 128): 57,
        (178, 178, 178, 178): 232,
    }
    synth_lut = {}

    def get_exact_tile(nw, ne, sw, se):
        key = (int(nw), int(ne), int(sw), int(se))
        if key in exact_lut:
            return exact_lut[key]
        if key in synth_lut:
            return synth_lut[key]
        flips = [
            ((nw, ne, sw, se), 0, 0),
            ((ne, nw, se, sw), 1, 0),
            ((sw, se, nw, ne), 0, 1),
            ((se, sw, ne, nw), 1, 1),
        ]
        for ck, fx, fy in flips:
            if ck in synth_lut and (synth_lut[ck] & 0xC000) == 0:
                tid = synth_lut[ck] & 0x0FFF
                val = tid | (fx << 15) | (fy << 14)
                synth_lut[key] = val
                return val
        tid = len(tile_bank)
        tile_bank.append(make_bilinear_tile(nw, ne, sw, se))
        synth_lut[key] = tid
        return tid

    for r, c in zip(*np.nonzero(walk)):
        hm_grid[r, c] = get_exact_tile(c00[r, c], c01[r, c], c10[r, c], c11[r, c])

    # Fix convex/diagonal cliff corners bordering walk so corner[0,0] strictly matches V[r,c]
    for r in range(1, D - 1):
        for c in range(1, W - 1):
            if walk[r, c]:
                continue
            touches_walk_corner = walk[r - 1, c - 1] or walk[r - 1, c] or walk[r, c - 1]
            touches_walk_any = (
                touches_walk_corner
                or walk[r, c + 1] or walk[r + 1, c]
                or walk[r + 1, c + 1] or walk[r - 1, c + 1] or walk[r + 1, c - 1]
            )
            if not touches_walk_any:
                continue
            v = int(hm_grid[r, c])
            t = tile_bank[v & 0x0FFF]
            if v & 0x8000:
                t = t[:, ::-1]
            if v & 0x4000:
                t = t[::-1, :]
            exp = (int(c00[r, c]), int(c01[r, c]), int(c10[r, c]), int(c11[r, c]))
            got = (int(t[0, 0]), int(t[0, 31]), int(t[31, 0]), int(t[31, 31]))
            if int(t[0, 0]) != int(c00[r, c]) or (kind[r, c] == KIND_CRAMP_L and got != exp):
                hm_grid[r, c] = get_exact_tile(*exp)

    path_layer = np.full((D, W), PATH_OPEN, dtype=np.uint16)
    path_layer[kind == KIND_CLIFF] = PATH_SOLID_ALL
    path_layer[kind == KIND_CRAMP_U] = PATH_SOLID_ALL
    path_layer[kind == KIND_CRAMP_L] = PATH_SLOPEFORCE
    hm_ids = hm_grid & 0x0FFF
    path_layer[np.isin(hm_ids, [166, 217, 180])] = PATH_SOLID_ALL

    return hm_grid, path_layer, np.array(tile_bank, dtype=np.uint8)
