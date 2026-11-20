#!/usr/bin/env python3
"""Canonical generator for the "Pangaea Basin" 1/4-scale custom Level 1.

Produces:
  Data/Terrain/Level1.ter, Data/Terrain/Level1Pro.ter (same geometry, Pro has
  scaled-up enemies & powerups matching original Level1Pro.ter ratios),
  Data/Terrain/Level1.trt (verbatim copy of the original tileset),
  Data/Images/Map.tga (GPS map, RLE colormapped 8-bit, matching TGA.c),
  and tools/level_preview.png (human-review render).

Design: An organic, asymmetric S-curve valley & basin world (~6,000-6,800
BFS-playable tiles) centered inside the 270x355 grid (roughly cols 80..190,
rows 110..248):
  - Zone 1: Emerald Marsh (NW) — lush grass, 2-3 organic lakes (hm=41) with
    a shallow-ford (hm=67) lake island, Start (S), Portal P0, 5x Species 0 eggs,
    Fern/Bamboo/Cypress/StickPalm groves, Mushrooms, Boulders, SporePods,
    Rex + Stego.
  - Zone 2: Obsidian Lava Gorge (NE) — dark volcanic ash/basalt, wide organic
    lava river/basin (hm=67) with FLOOR (hm=85) islands including a central
    Portal Island (P1), 2 StepStone bridges (parm=(0,0,0,1), spaced 3.2-3.5
    tiles apart over flat hm=67 lava with nearby LavaPatch parm3=0), 1 raised
    basalt overlook (hm=109 with smooth walkable ramp), 2 GasVents, 5x Species 1
    eggs, Fern/PinePalm, Bushes (parm3=1 Tricer ambushes), Ptera + Tricer + Rex.
  - Zone 3: Crystal Canyon & Oasis Crossroads (Central/W) — tan/ochre canyon
    with a Crystal Bowl (MAKEDUST sand + ~24 Crystals 0..2), a Palm Oasis
    (hm=41 pool), a western canyon loop connecting back to Emerald Marsh,
    Portal P2, RollBoulders on hm=255 canyon rims overlooking the trail,
    5x Species 2 eggs, StickPalm/MainPalm/Fern, SporePods,
    Spitter + Rex + Tricer + Ptera.
  - Zone 4: Ancient Nest Lagoon (South Endgame) — scenic endgame lake (hm=41)
    with Portal P3 on a hm=109 overlook terrace (walkable ramp), 4 raised
    Nest Terraces/Spires (hm=109/128 walkable ramps + hm=178 spires with
    GasVents beside them), 5x Species 3 eggs (on nest terraces/spires) +
    5x Species 4 eggs (on lagoon islands/shores), Bamboo/MainPalm/StickPalm/Fern,
    GasVents, PowerUps, mixed endgame enemies.

Engine fidelity:
  - Continuous vertex heightfield V[0..D, 0..W] where every tile (r,c) gets
    an hm_tile whose post-flip pixel[0,0] strictly equals V[r,c], eliminating
    vertex spikes and cracks.
  - Smooth multi-tile ramps on walkable path=0 (normal.y > 0.85; max step <= 20).
  - Clean cliff kit (original 166/167/180/181/182/217/235 + exact vertex-matched
    cliff skirt/wall tiles) with path=9 on CLIFF/CRAMP_U and path=43 on CRAMP_L.
  - 100% IT_WATER coverage of hm=41 and 100% IT_LAVA coverage of hm=67 lava.
  - TILE_ATTRIB_WATER only on hm<=47 water tiles; TILE_ATTRIB_LAVA only on hm=67 lava.

Deterministic: fixed RNG seed.
"""

from __future__ import annotations

import math
import shutil
import struct
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
ORIG_DIR = ROOT / "tools" / "original_terrain"
OUT_TERRAIN = ROOT / "Data" / "Terrain"
OUT_IMAGES = ROOT / "Data" / "Images"
PREVIEW_PATH = ROOT / "tools" / "level_preview.png"
BUNDLE = ROOT / "build" / "RelWithDebInfo" / "Nanosaur.app" / "Contents" / "Resources"

W, D = 270, 355           # map size in tiles (cols, rows)
TILE_PX = 32              # Oreo map pixels per tile
SEED = 20261120

# ---------------------------------------------------------------- item types
IT_START, IT_POWERUP, IT_TRICER, IT_REX, IT_LAVA = 0, 1, 2, 3, 4
IT_EGG, IT_GASVENT, IT_PTERA, IT_STEGO, IT_PORTAL = 5, 6, 7, 8, 9
IT_TREE, IT_BOULDER, IT_MUSHROOM, IT_BUSH, IT_WATER = 10, 11, 12, 13, 14
IT_CRYSTAL, IT_SPITTER, IT_STEPSTONE, IT_ROLLBOULDER, IT_SPOREPOD = 15, 16, 17, 18, 19
ENEMY_TYPES = (IT_TRICER, IT_REX, IT_PTERA, IT_STEGO, IT_SPITTER)

# tree parm0 (Items.c AddTree)
TREE_FERN, TREE_STICKPALM, TREE_BAMBOO, TREE_CYPRESS, TREE_MAINPALM, TREE_PINEPALM = range(6)

# powerup parm0 (Triggers.c AddPowerUp)
POW_HEAT, POW_LASER, POW_TRI, POW_HEALTH, POW_SHIELD, POW_NUKE, POW_SONIC = range(7)

# ------------------------------------------------------------- terrain kinds
KIND_CLIFF = 0
KIND_CRAMP_U = 1
KIND_CRAMP_L = 2
KIND_FLOOR = 3
KIND_WATER_BED = 4
KIND_WATER_SHELF = 5
KIND_FORD = 6
KIND_LAVA_BED = 7
KIND_TERRACE_109 = 8
KIND_TERRACE_128 = 9
KIND_SPIRE_178 = 10
KIND_WALK_RAMP = 11

# Zone IDs
ZONE_MARSH = 0
ZONE_LAVA = 1
ZONE_CANYON = 2
ZONE_LAGOON = 3

# ------------------------------------------------------------ texture weaves
# Verified against Level1.ter attributes and RGB averages:
MARSH_GRASS = [19, 20, 21, 38, 39]          # lush green grass
MARSH_SHORE = [74, 89]                      # muddy shore
WATER_MARSH_TEX = [23, 24, 42, 43]          # TILE_ATTRIB_WATER (0x80)
WATER_LAGOON_TEX = [223, 224, 231, 232]     # TILE_ATTRIB_WATER (0x80)

LAVA_ROCK = [148, 149, 164, 165, 170, 185, 186, 195, 207]  # dark volcanic basalt
LAVA_TEX = [137, 138, 151, 152, 0xC000 | 137, 0xC000 | 138, 0x8000 | 151, 0x4000 | 152]  # TILE_ATTRIB_LAVA (0x20)

CANYON_TEX = [189, 190, 200, 201, 132]      # tan/ochre canyon
CRYSTAL_SAND = [144, 145, 160, 161]         # TILE_ATTRIB_MAKEDUST (0x10)

LAGOON_SHORE = [19, 20, 21, 74, 89, 189, 190]
TERRACE_TEX = [189, 190, 200, 201, 132]
SPIRE_TEX = [148, 164, 185, 186]

CLIFF_PLATEAU_MARSH = [17, 36, 189, 190, 200, 201]
CLIFF_PLATEAU_LAVA = [148, 149, 164, 165, 207, 185]
CLIFF_PLATEAU_CANYON = [17, 36, 189, 190, 200, 201]
CLIFF_PLATEAU_LAGOON = [17, 36, 189, 200, 201]

# Original Level1 cliff kit (orth stacks + diagonal corners)
CLIFF_ORTH = {  # facing -> (lower_tid, upper_tid, fx, fy)
    "E": (167, 166, 0, 0),
    "W": (167, 166, 1, 0),
    "S": (235, 217, 0, 0),
    "N": (235, 217, 0, 1),
}
CLIFF_DIAG = {"SW": (0, 0), "SE": (1, 0), "NW": (0, 1), "NE": (1, 1)}
CLIFF_TEX = {  # (tid, fx, fy) -> full texture value from original Level1
    (167, 0, 0): 20561, (167, 0, 1): 20560, (167, 1, 0): 4176, (167, 1, 1): 4177,
    (166, 0, 0): 20545, (166, 0, 1): 20544, (166, 1, 0): 4160, (166, 1, 1): 4161,
    (235, 0, 0): 80, (235, 0, 1): 16464, (235, 1, 0): 32848, (235, 1, 1): 49232,
    (217, 0, 0): 64, (217, 0, 1): 16448, (217, 1, 0): 32832, (217, 1, 1): 49216,
    (181, 0, 0): 16506, (181, 0, 1): 66, (181, 1, 0): 49218, (181, 1, 1): 32890,
    (180, 0, 0): 16496, (180, 0, 1): 50, (180, 1, 0): 49264, (180, 1, 1): 32880,
    (182, 0, 0): 16516, (182, 0, 1): 132, (182, 1, 0): 49355, (182, 1, 1): 32971,
}
WALL_FALLBACK = [64, 65, 66, 122, 80, 81]

TILE_ATTRIB_MAKEDUST = 1 << 4
TILE_ATTRIB_LAVA = 1 << 5
TILE_ATTRIB_WATER = 1 << 7

PATH_OPEN = 0
PATH_SOLID_ALL = 9
PATH_SLOPEFORCE = 43


# =========================================================================
# Original data loading
# =========================================================================
class Original:
    def __init__(self):
        data = (ORIG_DIR / "Level1.ter").read_bytes()
        h = struct.unpack_from(">7i2h2i", data, 0)
        (tex_off, hm_off, path_off, obj_off, _, hmtile_off, _,
         ow, od, attrib_off, anim_off) = h
        assert (ow, od) == (W, D), "original size mismatch"
        n = ow * od
        self.tex = np.frombuffer(data, ">u2", n, tex_off).reshape(od, ow)
        self.hm = np.frombuffer(data, ">u2", n, hm_off).reshape(od, ow)
        self.path = np.frombuffer(data, ">u2", n, path_off).reshape(od, ow)
        n_hm_tiles = (attrib_off - hmtile_off) // 1024
        self.hm_tiles = np.frombuffer(
            data, "u1", n_hm_tiles * 1024, hmtile_off).reshape(n_hm_tiles, 32, 32)
        self.attrib_blob = data[attrib_off:anim_off]
        self.anim_blob = data[anim_off:]
        self.attribs = [struct.unpack_from(">Hhbbh", self.attrib_blob, i * 8)
                        for i in range(len(self.attrib_blob) // 8)]
        trt = (ORIG_DIR / "Level1.trt").read_bytes()
        n_tex = struct.unpack_from(">i", trt, 0)[0]
        texels = np.frombuffer(trt, ">u2", n_tex * 32 * 32, 4).reshape(n_tex, 32, 32)
        r = ((texels >> 10) & 31) * 255 // 31
        g = ((texels >> 5) & 31) * 255 // 31
        b = (texels & 31) * 255 // 31
        self.tile_avg = np.stack(
            [r.mean(axis=(1, 2)), g.mean(axis=(1, 2)), b.mean(axis=(1, 2))],
            axis=1).astype(np.uint8)
        self.n_tex_tiles = n_tex


# =========================================================================
# Geometry helpers
# =========================================================================
def dilate8(mask):
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
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def erode4(mask):
    out = mask.copy()
    out[1:, :] &= mask[:-1, :]
    out[:-1, :] &= mask[1:, :]
    out[:, 1:] &= mask[:, :-1]
    out[:, :-1] &= mask[:, 1:]
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


def carve_ramp_platform(V, target_mask, target_h, walkable_mask, base_h=85, max_step=18):
    """Raise/lower a terrace plateau onto V with smooth multi-tile ramps inside walkable_mask.

    Computes BFS tile distance within walkable_mask from target_mask and ramps linearly
    so adjacent tile steps never exceed max_step (keeping normal.y > 0.85).
    """
    steps_needed = max(1, math.ceil(abs(target_h - base_h) / max_step))
    dist = np.full((D, W), -1, dtype=np.int32)
    dq = deque()
    for r, c in zip(*np.nonzero(target_mask)):
        dist[r, c] = 0
        dq.append((int(r), int(c)))
    while dq:
        r, c = dq.popleft()
        nd = dist[r, c] + 1
        if nd > steps_needed:
            continue
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < D and 0 <= nc < W and walkable_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = nd
                dq.append((nr, nc))
    ramp_mask = (dist > 0) & (dist <= steps_needed)
    for r, c in zip(*np.nonzero(target_mask)):
        V[r, c] = target_h
    for r, c in zip(*np.nonzero(ramp_mask)):
        frac = 1.0 - (dist[r, c] / (steps_needed + 1.0))
        hval = int(round(base_h + (target_h - base_h) * frac))
        if target_h > base_h:
            V[r, c] = max(V[r, c], hval)
        else:
            V[r, c] = min(V[r, c], hval)
    return ramp_mask


def ensure_smooth_walkable_heightfield(V, walkable_mask, locked_mask, max_step=18):
    """Relax unlocked walkable vertices until every adjacent walkable pair has |dV| <= max_step."""
    for _ in range(40):
        changed = False
        for r, c in zip(*np.nonzero(walkable_mask & ~locked_mask)):
            vals = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < D and 0 <= nc < W and walkable_mask[nr, nc]:
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


# =========================================================================
# Level layout: "Pangaea Basin" — S-curve valley & basin world
# =========================================================================
def build_world(rng):
    cc, rr = np.meshgrid(np.arange(W), np.arange(D))
    noise = smooth_noise_field(cc, rr)

    # ---- Zone 1: Emerald Marsh (NW, ~1,700 tiles) ----
    # Roughly cols 82..126, rows 112..164
    z_marsh = (
        paint_blob(cc, rr, 102, 136, 20.5, 19.0, -15, warp=0.14 * noise)
        | paint_blob(cc, rr, 92, 152, 12.5, 13.0, 20, warp=0.12 * noise)
        | paint_blob(cc, rr, 114, 124, 12.5, 11.0, 10, warp=0.10 * noise)
    )

    # ---- Zone 2: Obsidian Lava Gorge (NE, ~1,600 tiles) ----
    # Roughly cols 132..186, rows 112..164
    z_lava = (
        paint_blob(cc, rr, 159, 137, 21.0, 17.5, 12, warp=0.12 * noise)
        | paint_blob(cc, rr, 143, 124, 11.5, 10.0, -18, warp=0.10 * noise)
        | paint_blob(cc, rr, 173, 153, 11.5, 11.5, 25, warp=0.10 * noise)
    )

    # ---- Zone 3: Crystal Canyon & Oasis Crossroads (Central/W, ~1,500 tiles) ----
    # S-curve sweeping SW from Lava Gorge / central crossroads, plus western canyon loop back to Marsh
    z_canyon = (
        paint_capsule(cc, rr, (165, 164), (144, 180), 7.5, 8.2, warp=0.8 * noise)
        | paint_blob(cc, rr, 132, 182, 14.5, 11.5, -15, warp=0.12 * noise)   # Crystal Bowl + Crossroads
        | paint_blob(cc, rr, 109, 189, 12.0, 10.5, 10, warp=0.10 * noise)   # Palm Oasis
        | paint_capsule(cc, rr, (99, 185), (93, 163), 5.2, 5.5, warp=0.7 * noise)  # Western canyon loop -> Marsh
    )

    # ---- Zone 4: Ancient Nest Lagoon (South Endgame, ~1,500 tiles) ----
    # Sweeping SE from Oasis/Crossroads into scenic southern lagoon, cols 115..178, rows 200..250
    z_lagoon = (
        paint_capsule(cc, rr, (121, 198), (132, 213), 6.5, 7.5, warp=0.6 * noise)
        | paint_blob(cc, rr, 148, 226, 21.6, 17.8, -8, warp=0.12 * noise)
        | paint_blob(cc, rr, 168, 218, 10.5, 10.0, 20, warp=0.10 * noise)
    )

    # ---- Main S-curve connectors ----
    conn_marsh_lava = paint_capsule(cc, rr, (121, 127), (138, 127), 5.2, 5.2, warp=0.5 * noise)
    conn_lava_canyon = paint_capsule(cc, rr, (171, 155), (162, 169), 5.0, 5.0, warp=0.4 * noise)
    conn_canyon_lagoon = paint_capsule(cc, rr, (130, 193), (136, 207), 5.2, 5.2, warp=0.4 * noise)

    walk = z_marsh | z_lava | z_canyon | z_lagoon | conn_marsh_lava | conn_lava_canyon | conn_canyon_lagoon

    # Carve organic rocky mesa pillars/peninsulas inside canyons so terrain feels hand-crafted
    rock_pillars = (
        paint_blob(cc, rr, 125, 160, 7.8, 11.0, 8)     # central ridge dividing Marsh & Lava from Canyon
        | paint_blob(cc, rr, 107, 168, 4.8, 5.8, -10)  # western canyon bend divider
        | paint_blob(cc, rr, 150, 198, 6.8, 5.8, 12)   # ridge between Crystal Crossroads & Lagoon east
    )
    walk &= ~rock_pillars

    # Containment margin
    walk[:6, :] = walk[-6:, :] = False
    walk[:, :6] = walk[:, -6:] = False

    # Clean isolated 1-tile pinholes / single-tile spurs for crisp cliff kit grammar
    for _ in range(2):
        n_card = (
            np.roll(walk, 1, 0).astype(int) + np.roll(walk, -1, 0).astype(int)
            + np.roll(walk, 1, 1).astype(int) + np.roll(walk, -1, 1).astype(int)
        )
        walk |= (~walk) & (n_card >= 3)
        walk &= (walk & (n_card >= 2))

    # Zone map with smooth organic noise-warped boundaries (no straight geometric seams!)
    warp_c = cc + 4.5 * noise
    warp_r = rr + 4.5 * np.roll(noise, 7, axis=0)
    d_m = (warp_c - 102) ** 2 + (warp_r - 137) ** 2
    d_l = (warp_c - 159) ** 2 + (warp_r - 137) ** 2
    d_c = (warp_c - 122) ** 2 + (warp_r - 183) ** 2
    d_g = (warp_c - 148) ** 2 + (warp_r - 224) ** 2
    zone = np.argmin(np.stack([d_m, d_l, d_c, d_g]), axis=0).astype(np.uint8)
    zone[z_marsh] = ZONE_MARSH
    zone[z_lava] = ZONE_LAVA
    zone[z_canyon] = ZONE_CANYON
    zone[z_lagoon] = ZONE_LAGOON
    # Smooth organic transition across Marsh<->Lava connector and Lava<->Canyon trail
    marsh_lava_split = 130.0 + 3.2 * noise
    zone[conn_marsh_lava & (cc < marsh_lava_split)] = ZONE_MARSH
    zone[conn_marsh_lava & (cc >= marsh_lava_split)] = ZONE_LAVA
    # Organic Canyon-Lagoon blend around rows 197..204
    canyon_lagoon_split = 201.0 + 2.8 * noise
    zone[(z_canyon | z_lagoon) & (rr >= canyon_lagoon_split) & (cc > 118)] = ZONE_LAGOON

    # Cliff bands around walkable basin
    kind = np.full((D, W), KIND_CLIFF, dtype=np.uint8)
    kind[walk] = KIND_FLOOR

    d1 = dilate8(walk) & ~walk
    d2 = dilate8(dilate8(walk)) & ~walk & ~d1
    kind[d1] = KIND_CRAMP_L
    kind[d2] = KIND_CRAMP_U

    # Safe interior floor helper (at least 2 tiles away from cliffs)
    cliff_near2 = dilate8(dilate8(~walk))

    # =====================================================================
    # ZONE FEATURES (Water lakes, Lava river/bridges, Terraces, Crystal Bowl)
    # =====================================================================
    V = np.full((D + 1, W + 1), 255, dtype=np.int32)

    # --- 1. Emerald Marsh Lakes (3 organic lakes, hm=41) + shallow ford island ---
    marsh_lake_1 = (
        paint_blob(cc, rr, 104, 139, 9.5, 8.0, 12, warp=0.14 * noise)
        & walk & ~cliff_near2
    )
    marsh_island = paint_blob(cc, rr, 104, 139, 2.7, 2.4, 15, warp=0.16 * noise) & marsh_lake_1
    marsh_ford = (
        paint_capsule(cc, rr, (95, 139), (102, 139), 1.4, 1.5, warp=0.35 * noise)
        & marsh_lake_1 & ~marsh_island
    )
    marsh_lake_1_water = marsh_lake_1 & ~marsh_island & ~marsh_ford

    marsh_lake_2 = (
        paint_blob(cc, rr, 90, 151, 4.5, 4.0, -20, warp=0.12 * noise)
        & walk & ~cliff_near2
    )
    marsh_lake_3 = (
        paint_blob(cc, rr, 114, 122, 4.0, 3.5, 15, warp=0.10 * noise)
        & walk & ~cliff_near2
    )

    # --- 3b. Palm Oasis in Zone 3 (around 109, 189) ---
    oasis_pool = (
        paint_blob(cc, rr, 109, 189, 4.8, 4.2, -12, warp=0.12 * noise)
        & walk & ~cliff_near2
    )

    # --- 4. Ancient Nest Lagoon Lake (around 148, 219) with 2 organic islands ---
    lagoon_lake = (
        paint_blob(cc, rr, 148, 219, 11.2, 6.0, -5, warp=0.13 * noise)
        & walk & ~cliff_near2
    )
    lagoon_island_w = paint_blob(cc, rr, 142, 219, 2.5, 2.3, -18, warp=0.15 * noise) & lagoon_lake
    lagoon_island_e = paint_blob(cc, rr, 154, 218, 2.5, 2.3, 22, warp=0.15 * noise) & lagoon_lake
    lagoon_ford_w = paint_capsule(cc, rr, (136, 219), (141, 219), 1.35, 1.4, warp=0.32 * noise) & lagoon_lake & ~lagoon_island_w
    lagoon_ford_e = paint_capsule(cc, rr, (155, 218), (160, 218), 1.35, 1.4, warp=0.32 * noise) & lagoon_lake & ~lagoon_island_e
    lagoon_water = lagoon_lake & ~lagoon_island_w & ~lagoon_island_e & ~lagoon_ford_w & ~lagoon_ford_e

    all_water_beds = marsh_lake_1_water | marsh_lake_2 | marsh_lake_3 | oasis_pool | lagoon_water
    all_fords = marsh_ford | lagoon_ford_w | lagoon_ford_e
    all_water_islands = marsh_island | lagoon_island_w | lagoon_island_e

    # --- 2. Obsidian Lava Gorge (NE): wide organic lava river/basin + Portal Island + 2 StepStone bridges ---
    lava_basin = (
        paint_blob(cc, rr, 159, 138, 17.5, 12.8, 10, warp=0.13 * noise)
        & z_lava & walk & ~cliff_near2
    )
    lava_portal_island = paint_blob(cc, rr, 156, 138, 3.8, 3.4, -14, warp=0.14 * noise) & walk
    lava_north_island = paint_blob(cc, rr, 155, 125, 3.1, 2.7, 18, warp=0.15 * noise) & walk
    lava_east_island = paint_blob(cc, rr, 174, 141, 3.1, 2.7, -20, warp=0.15 * noise) & walk

    lava_islands = lava_portal_island | lava_north_island | lava_east_island
    lava_bed = lava_basin & ~lava_islands
    bridge_a_channel = paint_capsule(cc, rr, (141, 138), (152, 138), 3.0, 3.0, warp=0.3 * noise) & walk & ~lava_islands
    bridge_b_channel = paint_capsule(cc, rr, (160, 140), (171, 141), 3.0, 3.0, warp=0.3 * noise) & walk & ~lava_islands
    lava_bed |= (bridge_a_channel | bridge_b_channel)
    lava_bed &= ~lava_islands

    # --- 2b. Raised Basalt Overlook (hm=109) in Zone 2 ---
    lava_overlook = paint_blob(cc, rr, 142, 122, 3.0, 2.6, 12, warp=0.10 * noise) & walk & ~lava_bed & ~cliff_near2

    # --- 3. Zone 3 Crystal Bowl (around 132, 182) ---
    crystal_bowl = paint_blob(cc, rr, 132, 182, 8.5, 6.5, -15, warp=0.14 * noise) & walk & ~cliff_near2

    # --- 4b. Zone 4 Nest Terraces/Spires in Ancient Nest Lagoon ---
    lagoon_p3_terrace = paint_blob(cc, rr, 128, 217, 2.8, 2.6, 8, warp=0.08 * noise) & walk & ~lagoon_lake & ~cliff_near2
    nest_terrace_a = paint_blob(cc, rr, 129, 234, 2.8, 2.5, -12, warp=0.08 * noise) & walk & ~lagoon_lake & ~cliff_near2
    nest_terrace_b = paint_blob(cc, rr, 164, 232, 2.8, 2.5, 14, warp=0.08 * noise) & walk & ~lagoon_lake & ~cliff_near2
    nest_terrace_c = paint_blob(cc, rr, 168, 215, 2.8, 2.4, -10, warp=0.08 * noise) & walk & ~lagoon_lake & ~dilate8(dilate8(dilate8(~walk)))
    # Two raised spires (hm=178) safely inside both cliff & lake skirt buffers so ramps step down <= 20px on ALL sides!
    cliff_near4 = dilate8(dilate8(dilate8(dilate8(~walk))))
    cliff_near5 = dilate8(cliff_near4)
    lake_near5 = dilate8(dilate8(dilate8(dilate8(dilate8(lagoon_lake)))))
    nest_spire_1 = paint_blob(cc, rr, 168, 223, 2.0, 2.0) & walk & ~lake_near5 & ~cliff_near4
    nest_spire_2 = paint_blob(cc, rr, 147, 237, 2.0, 2.0) & walk & ~lake_near5 & ~cliff_near5

    # Build continuous vertex heightfield V (size D+1 x W+1)
    walk_verts = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(walk)):
        walk_verts[r:r + 2, c:c + 2] = True

    V[:, :] = 255
    ring1_v = dilate8(walk_verts) & ~walk_verts
    V[ring1_v] = 149
    V[walk_verts] = 85

    def carve_vertex_ramp(tile_mask, target_h, walkable_tile_mask, base_h=85, max_step=18):
        v_target = np.zeros((D + 1, W + 1), dtype=bool)
        for r, c in zip(*np.nonzero(tile_mask)):
            v_target[r:r + 2, c:c + 2] = True
        v_walk = np.zeros((D + 1, W + 1), dtype=bool)
        for r, c in zip(*np.nonzero(walkable_tile_mask)):
            v_walk[r:r + 2, c:c + 2] = True

        steps_needed = max(1, math.ceil(abs(target_h - base_h) / max_step))
        dist = np.full((D + 1, W + 1), -1, dtype=np.int32)
        dq = deque()
        for r, c in zip(*np.nonzero(v_target)):
            dist[r, c] = 0
            V[r, c] = target_h
            dq.append((int(r), int(c)))
        while dq:
            r, c = dq.popleft()
            nd = dist[r, c] + 1
            if nd > steps_needed:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr <= D and 0 <= nc <= W and v_walk[nr, nc] and dist[nr, nc] < 0:
                    dist[nr, nc] = nd
                    dq.append((nr, nc))
        for r, c in zip(*np.nonzero((dist > 0) & (dist <= steps_needed))):
            frac = 1.0 - (dist[r, c] / (steps_needed + 1.0))
            hval = int(round(base_h + (target_h - base_h) * frac))
            if target_h > base_h:
                V[r, c] = max(V[r, c], hval)
            else:
                V[r, c] = min(V[r, c], hval)
        return v_target

    # Walkable domain allowed to slope for ramps (excludes lakes+4-tile shore, lava, and cliff border tiles)
    water_buffer = dilate8(dilate8(dilate8(dilate8(all_water_beds | all_fords | all_water_islands))))
    lava_buffer = dilate8(lava_bed | lava_islands)
    ramp_domain = walk & ~water_buffer & ~lava_buffer & ~dilate8(~walk)

    # Carve raised terraces (hm=109, 128) with gentle ramps (max_step=14..15 -> normal.y > 0.90)
    carve_vertex_ramp(lava_overlook, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(lagoon_p3_terrace, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(nest_terrace_a, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(nest_terrace_b, 128, ramp_domain, base_h=85, max_step=15)
    carve_vertex_ramp(nest_terrace_c, 128, ramp_domain, base_h=85, max_step=15)

    # Carve spires (hm=178) with smooth 6-step skirts (max_step=16 -> <=19px per tile on all sides!)
    carve_vertex_ramp(nest_spire_1, 178, ramp_domain, base_h=85, max_step=16)
    carve_vertex_ramp(nest_spire_2, 178, ramp_domain, base_h=85, max_step=16)

    # 2. Water basins in vertex space:
    cliff_v = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(~walk)):
        cliff_v[r:r + 2, c:c + 2] = True

    def carve_lake_vertices(bed_tile_mask, island_tile_mask=None, ford_tile_mask=None):
        vb = np.zeros((D + 1, W + 1), dtype=bool)
        for r, c in zip(*np.nonzero(bed_tile_mask)):
            vb[r:r + 2, c:c + 2] = True

        vi = np.zeros((D + 1, W + 1), dtype=bool)
        if island_tile_mask is not None and island_tile_mask.any():
            for r, c in zip(*np.nonzero(island_tile_mask)):
                vi[r:r + 2, c:c + 2] = True
        vf = np.zeros((D + 1, W + 1), dtype=bool)
        if ford_tile_mask is not None and ford_tile_mask.any():
            for r, c in zip(*np.nonzero(ford_tile_mask)):
                vf[r:r + 2, c:c + 2] = True

        no_41 = cliff_v | vi | dilate8(dilate8(vi)) | vf | dilate8(vf)
        vb = vb & walk_verts & ~no_41

        v_47 = dilate8(vb) & ~vb & walk_verts & ~cliff_v & ~vi & ~vf
        v_67 = dilate8(vb | v_47) & ~(vb | v_47) & walk_verts & ~cliff_v & ~vi
        v_67 |= ((vf | (dilate8(vi) & ~vi)) & walk_verts & ~cliff_v)

        V[v_67] = np.minimum(V[v_67], 67)
        V[v_47] = np.minimum(V[v_47], 47)
        V[vb] = 41
        V[(dilate8(vi) & ~vi) & walk_verts & ~cliff_v & (V < 67)] = 67
        V[vi & walk_verts] = 85

    carve_lake_vertices(marsh_lake_1_water, marsh_island, marsh_ford)
    carve_lake_vertices(marsh_lake_2)
    carve_lake_vertices(marsh_lake_3)
    carve_lake_vertices(oasis_pool)
    carve_lake_vertices(lagoon_water, lagoon_island_w | lagoon_island_e, lagoon_ford_w | lagoon_ford_e)

    # 3. Lava basin in vertex space:
    v_lava = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(lava_bed)):
        v_lava[r:r + 2, c:c + 2] = True
    v_lava_isl = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(lava_islands)):
        v_lava_isl[r:r + 2, c:c + 2] = True
    v_lava &= walk_verts & ~cliff_v & ~v_lava_isl
    V[v_lava] = 67
    V[v_lava_isl & walk_verts] = 85

    # Lock cliff boundary vertices & feature plateaus, then smooth any residual >20px steps everywhere!
    v_locked = cliff_v | (V <= 67) | v_lava_isl
    for m in (marsh_island, lagoon_island_w, lagoon_island_e,
              lava_overlook, lagoon_p3_terrace, nest_terrace_a, nest_terrace_b, nest_terrace_c,
              nest_spire_1, nest_spire_2):
        for r, c in zip(*np.nonzero(m)):
            v_locked[r:r + 2, c:c + 2] = True

    ensure_smooth_walkable_heightfield(V, walk_verts, v_locked, max_step=20)

    # 4. Compute per-tile height stats from V[r..r+1, c..c+1]
    c00 = V[:-1, :-1]
    c01 = V[:-1, 1:]
    c10 = V[1:, :-1]
    c11 = V[1:, 1:]
    tmax = np.maximum.reduce([c00, c01, c10, c11])
    tmin = np.minimum.reduce([c00, c01, c10, c11])
    tmean = 0.25 * (c00 + c01 + c10 + c11)

    # Classify tiles accurately from V
    is_water_bed = walk & (tmax == 41) & (tmin == 41)
    is_water_47 = walk & (tmax <= 47) & ~is_water_bed
    is_water_shore = walk & (tmin <= 47) & (tmax > 47)

    # Lava tiles: flat hm=67 inside lava zone (and NOT marsh/lagoon fords!)
    is_lava_tile = walk & (zone == ZONE_LAVA) & (tmax == 67) & (tmin == 67)
    is_ford_tile = walk & (zone != ZONE_LAVA) & (tmax == 67) & (tmin == 67)
    is_lava_rim = walk & (zone == ZONE_LAVA) & (tmin == 67) & (tmax == 85)

    # Classify terraces & ramps
    kind[is_water_bed] = KIND_WATER_BED
    kind[is_water_47 | is_water_shore] = KIND_WATER_SHELF
    kind[is_ford_tile] = KIND_FORD
    kind[is_lava_tile] = KIND_LAVA_BED
    kind[walk & (tmin == 109) & (tmax == 109)] = KIND_TERRACE_109
    kind[walk & (tmin == 128) & (tmax == 128)] = KIND_TERRACE_128
    kind[walk & (tmin == 178) & (tmax == 178)] = KIND_SPIRE_178
    kind[walk & (kind == KIND_FLOOR) & ((tmin != 85) | (tmax != 85))] = KIND_WALK_RAMP

    return {
        "V": V,
        "walk": walk,
        "kind": kind,
        "zone": zone,
        "c00": c00, "c01": c01, "c10": c10, "c11": c11,
        "tmax": tmax, "tmin": tmin, "tmean": tmean,
        "is_water_bed": is_water_bed,
        "is_water_47": is_water_47,
        "is_water_shore": is_water_shore,
        "is_lava_tile": is_lava_tile,
        "is_lava_rim": is_lava_rim,
        "is_ford_tile": is_ford_tile,
        "marsh_island": marsh_island,
        "lagoon_island_w": lagoon_island_w,
        "lagoon_island_e": lagoon_island_e,
        "lava_portal_island": lava_portal_island,
        "lava_north_island": lava_north_island,
        "lava_east_island": lava_east_island,
        "lava_overlook": lava_overlook,
        "crystal_bowl": crystal_bowl,
        "lagoon_p3_terrace": lagoon_p3_terrace,
        "nest_terrace_a": nest_terrace_a,
        "nest_terrace_b": nest_terrace_b,
        "nest_terrace_c": nest_terrace_c,
        "nest_spire_1": nest_spire_1,
        "nest_spire_2": nest_spire_2,
        "rock_pillars": rock_pillars,
    }


# =========================================================================
# Heightmap tile synthesis & cliff kit stamping
# =========================================================================
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
    # Guarantee exact corners
    arr[0, 0] = int(nw)
    arr[0, 31] = int(ne)
    arr[31, 0] = int(sw)
    arr[31, 31] = int(se)
    return arr


def stamp_cliff_kit(kind, walk):
    """Stamp original Level1 cliff grammar on CRAMP_L / CRAMP_U / CLIFF, WITHOUT touching walkable floor!

    Crucial fix vs old generator: NEVER stamp 182 (which has a 145 spike) onto walkable path=0 tiles!
    Instead, keep all walkable tiles strictly matching V[r,c], and only stamp cliff kit on
    CLIFF / CRAMP_U / CRAMP_L tiles.
    """
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

    # Pass 1: diagonals (181 on CRAMP_L, 180 on CRAMP_U/CLIFF; NO 182 on walkable floor!)
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
    V = world["V"]
    c00, c01, c10, c11 = world["c00"], world["c01"], world["c10"], world["c11"]

    hm_grid, cliff_assigned = stamp_cliff_kit(kind, walk)

    tile_bank = list(orig.hm_tiles)  # 0..247

    # Prefer known clean flat tiles from original
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
        # Share synthesized tiles across all 4 flip symmetries!
        # Let base tile have corners ck = (a, b, c, d) = (NW, NE, SW, SE).
        # Applying flip (fx, fy) produces:
        #   (0, 0) -> (a, b, c, d)
        #   (1, 0) -> (b, a, d, c)
        #   (0, 1) -> (c, d, a, b)
        #   (1, 1) -> (d, c, b, a)
        # Since each flip is an involution, ck = apply_flip(key, fx, fy)!
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

    # Fill all walkable tiles with exact-corner bilinear tiles!
    for r, c in zip(*np.nonzero(walk)):
        hm_grid[r, c] = get_exact_tile(c00[r, c], c01[r, c], c10[r, c], c11[r, c])

    # Fix convex/diagonal cliff corners bordering walk:
    # Any non-walk tile (r, c) whose corner (0, 0) belongs to a walk tile's mesh quad
    # (or any CRAMP_L tile where len(lows)==0 / diagonal 181 where corner[0,0] != V[r,c])
    # must have post-flip corner[0,0] == V[r,c] == 85 (never 149/170/250 spikes!).
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
            # If corner[0,0] mismatches V[r,c], OR if it's a diagonal 181/CRAMP_L touching walk
            # whose corners don't match continuous V (85 on walk corners, 149/255 on cliff corners):
            exp = (int(c00[r, c]), int(c01[r, c]), int(c10[r, c]), int(c11[r, c]))
            got = (int(t[0, 0]), int(t[0, 31]), int(t[31, 0]), int(t[31, 31]))
            if int(t[0, 0]) != int(c00[r, c]) or (kind[r, c] == KIND_CRAMP_L and got != exp):
                hm_grid[r, c] = get_exact_tile(*exp)

    # Path layer
    path_layer = np.full((D, W), PATH_OPEN, dtype=np.uint16)
    path_layer[kind == KIND_CLIFF] = PATH_SOLID_ALL
    path_layer[kind == KIND_CRAMP_U] = PATH_SOLID_ALL
    path_layer[kind == KIND_CRAMP_L] = PATH_SLOPEFORCE
    hm_ids = hm_grid & 0x0FFF
    path_layer[np.isin(hm_ids, [166, 217, 180])] = PATH_SOLID_ALL

    return hm_grid, path_layer, np.array(tile_bank, dtype=np.uint8)


# =========================================================================
# Texture layer (strict hazard attribute containment)
# =========================================================================
def paint_textures(world, hm_grid, orig, rng):
    kind = world["kind"]
    zone = world["zone"]
    walk = world["walk"]
    tmax = world["tmax"]
    is_water_bed = world["is_water_bed"]
    is_water_47 = world["is_water_47"]
    is_water_shore = world["is_water_shore"]
    is_lava_tile = world["is_lava_tile"]
    is_lava_rim = world["is_lava_rim"]
    is_ford_tile = world["is_ford_tile"]
    crystal_bowl = world["crystal_bowl"]

    tex = np.zeros((D, W), dtype=np.uint16)

    def pick(opts):
        return opts[rng.randrange(len(opts))]

    # Map cliff kit textures from original
    for r in range(D):
        for c in range(W):
            v = int(hm_grid[r, c])
            key = (v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1)
            k = kind[r, c]
            z = zone[r, c]

            if not walk[r, c]:
                if key in CLIFF_TEX:
                    tex[r, c] = CLIFF_TEX[key]
                elif key[0] in (166, 167, 180, 181, 217, 235):
                    tex[r, c] = pick(WALL_FALLBACK)
                else:
                    # High cliff plateau (hm=255)
                    if z == ZONE_MARSH:
                        tex[r, c] = pick(CLIFF_PLATEAU_MARSH)
                    elif z == ZONE_LAVA:
                        tex[r, c] = pick(CLIFF_PLATEAU_LAVA)
                    elif z == ZONE_CANYON:
                        tex[r, c] = pick(CLIFF_PLATEAU_CANYON)
                    else:
                        tex[r, c] = pick(CLIFF_PLATEAU_LAGOON)
                continue

            # Walkable tiles: STRICT hazard containment!
            # 1. Underwater (tmax <= 47): TILE_ATTRIB_WATER
            if is_water_bed[r, c] or is_water_47[r, c]:
                tex[r, c] = pick(WATER_LAGOON_TEX if z == ZONE_LAGOON else WATER_MARSH_TEX)
            # 2. Lava bed (flat hm=67 in Zone 2): TILE_ATTRIB_LAVA
            elif is_lava_tile[r, c]:
                tex[r, c] = pick(LAVA_TEX)
            # 3. Water shore / ford (dry or ankle-high bank, NEVER TILE_ATTRIB_WATER!)
            elif is_water_shore[r, c] or is_ford_tile[r, c]:
                tex[r, c] = pick(MARSH_SHORE)
            # 4. Lava rim / Zone 2 volcanic basalt (NEVER TILE_ATTRIB_LAVA!)
            elif z == ZONE_LAVA:
                if k == KIND_TERRACE_109 or k == KIND_WALK_RAMP:
                    tex[r, c] = pick([148, 149, 164, 165, 207])
                else:
                    tex[r, c] = pick(LAVA_ROCK)
            # 5. Zone 1 Emerald Marsh
            elif z == ZONE_MARSH:
                if rng.random() < 0.08:
                    tex[r, c] = pick(MARSH_SHORE)
                else:
                    tex[r, c] = pick(MARSH_GRASS)
            # 6. Zone 3 Crystal Canyon & Oasis Crossroads
            elif z == ZONE_CANYON:
                if crystal_bowl[r, c] and k == KIND_FLOOR:
                    tex[r, c] = pick(CRYSTAL_SAND)
                else:
                    tex[r, c] = pick(CANYON_TEX)
            # 7. Zone 4 Ancient Nest Lagoon
            else:
                if k in (KIND_TERRACE_109, KIND_TERRACE_128, KIND_WALK_RAMP):
                    tex[r, c] = pick(TERRACE_TEX)
                elif k == KIND_SPIRE_178:
                    tex[r, c] = pick(SPIRE_TEX)
                else:
                    tex[r, c] = pick(LAGOON_SHORE)

    return tex


# =========================================================================
# Item placement
# =========================================================================
def px(col, row):
    return col * TILE_PX + 16, row * TILE_PX + 16


class Placer:
    def __init__(self, rng, bfs_set, world):
        self.rng = rng
        self.bfs = bfs_set
        self.world = world
        self.items = []                     # (x, z, type, (p0,p1,p2,p3), pro_only)
        self.occupied = set()
        self.protected = []                 # (col, row) with 3-tile clearance
        self.by_cat = {"enemy": [], "powerup": [], "tree": [], "scenery": []}

    def cell_free(self, col, row, cat, nn):
        if (col, row) in self.occupied:
            return False
        for pc, pr in self.protected:
            if (col - pc) ** 2 + (row - pr) ** 2 < 3 ** 2:
                return False
        if cat in self.by_cat and nn > 0:
            for oc, orow in self.by_cat[cat]:
                if (col - oc) ** 2 + (row - orow) ** 2 < nn * nn:
                    return False
        return True

    def add(self, col, row, itype, parms=(0, 0, 0, 0), cat="scenery", nn=1.5,
            pro_only=False, protect=False):
        col, row = int(col), int(row)
        x, z = px(col, row)
        self.items.append((x, z, itype, tuple(parms), pro_only))
        self.occupied.add((col, row))
        if cat in self.by_cat:
            self.by_cat[cat].append((col, row))
        if protect:
            self.protected.append((col, row))

    def add_exact(self, x_px, z_px, itype, parms=(0, 0, 0, 0), cat="special",
                  pro_only=False, protect=True):
        """Add an item at sub-tile Oreo pixel coordinates (e.g., StepStone bridges)."""
        col, row = int(x_px) // TILE_PX, int(z_px) // TILE_PX
        self.items.append((int(round(x_px)), int(round(z_px)), itype, tuple(parms), pro_only))
        self.occupied.add((col, row))
        if protect:
            self.protected.append((col, row))

    def scatter(self, n, cells, itype, parm_fn, cat, nn, pro_only=False):
        placed = 0
        attempts = 0
        cells = list(cells)
        while placed < n and attempts < max(200, n * 600) and cells:
            attempts += 1
            row, col = cells[self.rng.randrange(len(cells))]
            if (row, col) not in self.bfs:
                continue
            if not self.cell_free(col, row, cat, nn):
                continue
            self.add(col, row, itype, parm_fn(placed), cat, pro_only=pro_only)
            placed += 1
        return placed


def nearest_dry_cell(placer, world, col, row, allowed_mask=None, radius=16, avoid=(), clearance=0):
    best, bd = None, 1e18
    for r in range(max(0, row - radius), min(D, row + radius + 1)):
        for c in range(max(0, col - radius), min(W, col + radius + 1)):
            if (r, c) not in placer.bfs:
                continue
            if allowed_mask is not None and not allowed_mask[r, c]:
                continue
            if not placer.cell_free(c, r, "special", 0):
                continue
            if any((c - ac) ** 2 + (r - ar) ** 2 < clearance ** 2 for ac, ar in avoid):
                continue
            d = (c - col) ** 2 + (r - row) ** 2
            if d < bd:
                bd, best = d, (c, r)
    if best is None:
        raise SystemExit(f"no dry cell near ({col},{row})")
    return best


def compute_patch_centers(mask, min_gap=8):
    """Cover 100% of `mask` tiles using 8x8 tile patches centered at (col, row).

    Each 8x8 patch at (c, r) covers [c-4, c+4) x [r-4, r+4) in tile indices
    (since patch center is tile center c+0.5, r+0.5, spanning +-4 tiles = [c-3.5, c+4.5]).
    Greedy set cover over valid mask centers guarantees 100% coverage of `mask`!
    """
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not cells:
        return []
    uncovered = set(cells)
    # Candidate centers: prefer interior mask cells so patches overlap minimal cliff/land,
    # while allowing any mask cell if needed for 100% coverage.
    chosen = []
    # Also seed a coarse grid of centers every 7 tiles first for neat alignment
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    r1 = max(r for r, _ in cells)
    c1 = max(c for _, c in cells)
    cell_set = set(cells)

    def covered_by(cr, cc):
        return {(r, c) for r in range(cr - 3, cr + 5) for c in range(cc - 3, cc + 5) if (r, c) in uncovered}

    def covered_by_wide(cr, cc):
        # Note: in Terrain/Items.c, patch center is at (cc+0.5, cr+0.5) * 140, scale=2.0 -> 8x8 tiles:
        # [cc-3.5, cc+4.5] in x and [cr-3.5, cr+4.5] in z -> tile centers c+0.5 in [cc-3.5, cc+4.5] -> c in [cc-4, cc+4]!
        # Wait: let's be conservative and require every mask tile (r, c) to lie in [cr-3, cr+4] x [cc-3, cc+4]
        # (which is strictly inside [cc-3.5, cc+4.5]!).
        return {(r, c) for r in range(cr - 3, cr + 5) for c in range(cc - 3, cc + 5) if (r, c) in uncovered}

    while uncovered:
        best_pt = None
        best_cov = set()
        # Evaluate candidate centers near remaining uncovered cells
        sample = list(uncovered)
        # Prefer centers that cover the most uncovered tiles and are well inside mask
        candidates = set()
        for ur, uc in sample:
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    pt = (ur + dr, uc + dc)
                    if pt in cell_set:
                        candidates.add(pt)
        if not candidates:
            candidates = set(sample)
        for cr, cc in candidates:
            if any(abs(cr - ar) < min_gap - 3 and abs(cc - ac) < min_gap - 3 for ar, ac in chosen):
                # allow closer only if needed to cover edge tiles
                pass
            cov = covered_by_wide(cr, cc)
            if len(cov) > len(best_cov):
                best_cov = cov
                best_pt = (cr, cc)
        if not best_pt or not best_cov:
            # Fallback: pick any uncovered cell directly
            best_pt = next(iter(uncovered))
            best_cov = covered_by_wide(*best_pt) | {best_pt}
        chosen.append(best_pt)
        uncovered -= best_cov

    return chosen


def build_items(rng, world, hm_grid, path_layer):
    walk = world["walk"]
    kind = world["kind"]
    zone = world["zone"]
    V = world["V"]
    is_water_bed = world["is_water_bed"]
    is_water_47 = world["is_water_47"]
    is_water_shore = world["is_water_shore"]
    is_lava_tile = world["is_lava_tile"]
    is_lava_rim = world["is_lava_rim"]
    is_ford_tile = world["is_ford_tile"]

    # Walkable tiles for BFS from Start:
    # path == 0 and vertex height <= 185 (so hm=178 spires are walkable/BFS if ramped, and hm=255 cliffs are excluded)
    walk_open = (path_layer == PATH_OPEN) & (world["tmax"] <= 185)

    # Start in a safe grassy clearing in Zone 1 Emerald Marsh (NW, around col 88, row 135)
    start_c, start_r = 88, 134
    # Snap to flat hm=85 Marsh floor
    dry_marsh = walk_open & (zone == ZONE_MARSH) & (kind == KIND_FLOOR) & ~dilate8(is_water_bed | is_water_47 | ~walk)
    best_s = None
    bd = 1e18
    for r, c in np.argwhere(dry_marsh):
        d = (c - start_c) ** 2 + (r - start_r) ** 2
        if d < bd:
            bd, best_s = d, (int(r), int(c))
    sr, sc = best_s

    # BFS from Start
    bfs = set()
    dq = deque([(sr, sc)])
    bfs.add((sr, sc))
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < D and 0 <= nc < W and (nr, nc) not in bfs and walk_open[nr, nc]:
                bfs.add((nr, nc))
                dq.append((nr, nc))

    P = Placer(rng, bfs, world)

    # Dry walkable cells per zone (excludes water beds, underwater shelves, lava beds, and cliff rims)
    dry_mask = walk_open & ~is_water_bed & ~is_water_47 & ~is_lava_tile
    safe_dry_mask = dry_mask & ~dilate8(~walk) & ~is_water_shore & ~is_lava_rim

    zone_dry = [[] for _ in range(4)]
    zone_flat = [[] for _ in range(4)]
    for r, c in bfs:
        if safe_dry_mask[r, c]:
            zone_dry[zone[r, c]].append((r, c))
            if kind[r, c] == KIND_FLOOR:
                zone_flat[zone[r, c]].append((r, c))

    # ---------------- Start (S) ----------------
    # Aim East toward Marsh Lake & Lava Gorge
    P.add(sc, sr, IT_START, (6, 0, 0, 0), "special", protect=True)

    # ---------------- Water & Lava Patches (100% coverage!) ----------------
    # Water patches cover all hm=41 & hm=47 underwater tiles
    water_need = is_water_bed | is_water_47
    water_centers = compute_patch_centers(water_need, min_gap=7)
    for r, c in water_centers:
        P.add(c, r, IT_WATER, (0, 0, 0, 0), "special", protect=False)

    # StepStone Bridges in Zone 2 Obsidian Lava Gorge:
    # Build two StepStone bridges (parm=(0,0,0,1) reincarnating, spaced 3.2-3.5 tiles apart)
    # over flat hm=67 lava tiles, keeping nearby LavaPatch parm3=0 (no fireballs camping bridges!).
    bridge_points = []
    # Bridge 1: West lava shore -> Portal Island along row 138 (4 stones spaced ~3.27 tiles)
    row_b1 = 138
    lava_cols_b1 = [c for c in range(138, 156) if is_lava_tile[row_b1, c]]
    if len(lava_cols_b1) >= 4:
        c_start = min(lava_cols_b1) + 1.0
        c_end = max(lava_cols_b1) - 0.2
        span = c_end - c_start
        n_stones = max(3, math.ceil(span / 3.45) + 1)
        for i in range(n_stones):
            cf = c_start + (span * i / max(1, n_stones - 1))
            if is_lava_tile[row_b1, int(cf)]:
                bridge_points.append((cf, float(row_b1) + 0.5))

    # Bridge 2: Portal Island -> East Lava Island along row 140 (3 stones spaced ~3.35 tiles)
    row_b2 = 140
    lava_cols_b2 = [c for c in range(158, 175) if is_lava_tile[row_b2, c]]
    if len(lava_cols_b2) >= 4:
        c_start = min(lava_cols_b2) + 1.5
        c_end = max(lava_cols_b2) - 0.8
        span = c_end - c_start
        n_stones = 3 if span <= 7.2 else max(3, math.ceil(span / 3.45) + 1)
        for i in range(n_stones):
            cf = c_start + (span * i / max(1, n_stones - 1))
            if is_lava_tile[row_b2, int(cf)]:
                bridge_points.append((cf, float(row_b2) + 0.5))

    for cf, rf in bridge_points:
        P.add_exact(cf * TILE_PX, rf * TILE_PX, IT_STEPSTONE, (0, 0, 0, 1), "special", protect=True)

    # Lava patches cover 100% of hm=67 lava tiles; parm3=0 near bridges, parm3=2 elsewhere
    # Pass exclude_occupied or verify coverage after shifting off StepStone cells!
    lava_centers = compute_patch_centers(is_lava_tile, min_gap=7)
    lava_cov = np.zeros((D, W), dtype=bool)
    for r, c in lava_centers:
        if (c, r) in P.occupied:
            # Find adjacent lava cell that still covers (r, c) and is unoccupied
            alt = None
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < D and 0 <= nc < W and is_lava_tile[nr, nc] and (nc, nr) not in P.occupied:
                        alt = (nr, nc)
                        break
                if alt:
                    break
            if alt:
                r, c = alt
        if (c, r) not in P.occupied:
            near_bridge = any((c + 0.5 - bc) ** 2 + (r + 0.5 - br) ** 2 <= 9.0 ** 2 for bc, br in bridge_points)
            parm3 = 0 if near_bridge else 2
            P.add(c, r, IT_LAVA, (0, 0, 0, parm3), "special", protect=False)
            lava_cov[max(0, r - 3):min(D, r + 5), max(0, c - 3):min(W, c + 5)] = True
    # Cover any remaining uncovered lava tiles if a shifted center left an edge exposed
    while True:
        rem = np.argwhere(is_lava_tile & ~lava_cov)
        if len(rem) == 0:
            break
        ur, uc = int(rem[0, 0]), int(rem[0, 1])
        placed_extra = False
        for dr in range(-3, 5):
            for dc in range(-3, 5):
                cr, cc_ = ur - dr, uc - dc
                if 0 <= cr < D and 0 <= cc_ < W and is_lava_tile[cr, cc_] and (cc_, cr) not in P.occupied:
                    near_bridge = any((cc_ + 0.5 - bc) ** 2 + (cr + 0.5 - br) ** 2 <= 9.0 ** 2 for bc, br in bridge_points)
                    P.add(cc_, cr, IT_LAVA, (0, 0, 0, 0 if near_bridge else 2), "special", protect=False)
                    lava_cov[max(0, cr - 3):min(D, cr + 5), max(0, cc_ - 3):min(W, cc_ + 5)] = True
                    placed_extra = True
                    break
            if placed_extra:
                break
        if not placed_extra:
            break

    # ---------------- Portals (P0..P3) ----------------
    # P0: Zone 1 overlooking main marsh lake (around 98, 128)
    p0_c, p0_r = nearest_dry_cell(P, world, 99, 128, allowed_mask=dry_mask, radius=14)
    P.add(p0_c, p0_r, IT_PORTAL, (0, 0, 0, 0), "special", protect=True)

    # P1: Zone 2 central Portal Island in the lava basin (around 156, 138)
    p1_mask = world["lava_portal_island"] & dry_mask
    p1_c, p1_r = nearest_dry_cell(P, world, 156, 138, allowed_mask=p1_mask, radius=8)
    P.add(p1_c, p1_r, IT_PORTAL, (1, 0, 0, 0), "special", protect=True)

    # P2: Zone 3 Crystal Canyon / Oasis Crossroads (around 120, 185)
    p2_c, p2_r = nearest_dry_cell(P, world, 120, 185, allowed_mask=dry_mask, radius=12)
    P.add(p2_c, p2_r, IT_PORTAL, (2, 0, 0, 0), "special", protect=True)

    # P3: Zone 4 Ancient Nest Lagoon on hm=109 overlook terrace (around 128, 217)
    p3_mask = (kind == KIND_TERRACE_109) & world["lagoon_p3_terrace"]
    p3_c, p3_r = nearest_dry_cell(P, world, 128, 217, allowed_mask=p3_mask, radius=6)
    P.add(p3_c, p3_r, IT_PORTAL, (3, 0, 0, 0), "special", protect=True)

    # ---------------- Eggs (25 total: 5 per species 0..4) ----------------
    egg_avoid = [(sc, sr), (p0_c, p0_r), (p1_c, p1_r), (p2_c, p2_r), (p3_c, p3_r)]

    def place_egg(species, col, row, mask=None, radius=14, min_sep=10):
        c, r = nearest_dry_cell(
            P, world, col, row,
            allowed_mask=(mask if mask is not None else safe_dry_mask),
            radius=radius,
            avoid=egg_avoid,
            clearance=min_sep,
        )
        P.add(c, r, IT_EGG, (species, 0, 0, 1), "special", protect=True)
        egg_avoid.append((c, r))
        return c, r

    # Species 0 (5 eggs in Emerald Marsh, including 1 on the marsh lake island!)
    place_egg(0, 104, 139, mask=world["marsh_island"] & dry_mask, radius=5, min_sep=4)
    place_egg(0, 88, 156, min_sep=9)
    place_egg(0, 100, 152, min_sep=9)
    place_egg(0, 117, 131, min_sep=9)
    place_egg(0, 107, 121, min_sep=9)

    # Species 1 (5 eggs in Obsidian Lava Gorge, across islands/overlook/shores)
    place_egg(1, 155, 125, mask=world["lava_north_island"] & dry_mask, radius=5, min_sep=6)
    place_egg(1, 174, 141, mask=world["lava_east_island"] & dry_mask, radius=5, min_sep=6)
    place_egg(1, 142, 122, mask=world["lava_overlook"] & (kind == KIND_TERRACE_109), radius=5, min_sep=6)
    place_egg(1, 138, 144, min_sep=8)
    place_egg(1, 173, 157, min_sep=8)

    # Species 2 (5 eggs in Crystal Canyon & Oasis Crossroads)
    place_egg(2, 136, 177, min_sep=9)
    place_egg(2, 127, 190, min_sep=9)
    place_egg(2, 102, 192, min_sep=9)
    place_egg(2, 93, 170, min_sep=9)
    place_egg(2, 158, 170, min_sep=9)

    # Species 3 (5 eggs in Ancient Nest Lagoon on nest terraces & spires!)
    place_egg(3, 129, 234, mask=world["nest_terrace_a"] & (kind == KIND_TERRACE_109), radius=6, min_sep=6)
    place_egg(3, 164, 232, mask=world["nest_terrace_b"] & (kind == KIND_TERRACE_128), radius=6, min_sep=6)
    place_egg(3, 168, 215, mask=world["nest_terrace_c"] & (kind == KIND_TERRACE_128), radius=6, min_sep=6)
    place_egg(3, 168, 223, mask=world["nest_spire_1"] & (kind == KIND_SPIRE_178), radius=5, min_sep=6)
    place_egg(3, 147, 237, mask=world["nest_spire_2"] & (kind == KIND_SPIRE_178), radius=5, min_sep=6)

    # Species 4 (5 eggs in Ancient Nest Lagoon on lagoon islands & shores!)
    place_egg(4, 142, 219, mask=world["lagoon_island_w"] & dry_mask, radius=5, min_sep=6)
    place_egg(4, 154, 218, mask=world["lagoon_island_e"] & dry_mask, radius=5, min_sep=6)
    place_egg(4, 138, 210, min_sep=8)
    place_egg(4, 156, 210, min_sep=8)
    place_egg(4, 168, 226, min_sep=8)

    # ---------------- GasVents ----------------
    # 2 GasVents in Obsidian Lava Gorge (Zone 2) + 3 beside Zone 4 hm=178 spires
    gv_spots = [
        (147, 126, (kind == KIND_FLOOR) & (zone == ZONE_LAVA)),
        (162, 153, (kind == KIND_FLOOR) & (zone == ZONE_LAVA)),
        (165, 224, dry_mask & (zone == ZONE_LAGOON)),
        (143, 235, dry_mask & (zone == ZONE_LAGOON)),
        (154, 237, dry_mask & (zone == ZONE_LAGOON)),
    ]
    for gc, gr, gmask in gv_spots:
        c, r = nearest_dry_cell(P, world, gc, gr, allowed_mask=gmask, radius=8)
        P.add(c, r, IT_GASVENT, (0, 0, 0, 0), "special", protect=True)

    # ---------------- RollBoulders on hm=255 canyon rims overlooking Zone 3 trail ----------------
    # Note: CRAMP_L is d1 (1 dilation), CRAMP_U is d2 (2 dilations), so KIND_CLIFF starts at 3 dilations!
    rim_mask = (
        (kind == KIND_CLIFF)
        & dilate8(dilate8(dilate8((zone == ZONE_CANYON) & walk)))
        & ~dilate8(dilate8(walk))
    )
    rim_cells = [(int(r), int(c)) for r, c in np.argwhere(rim_mask)]
    rng.shuffle(rim_cells)
    rb_placed = []
    for r, c in rim_cells:
        if len(rb_placed) >= 7:
            break
        if any((r - pr) ** 2 + (c - pc) ** 2 < 9 ** 2 for pr, pc in rb_placed):
            continue
        if 8 <= r < D - 8 and 8 <= c < W - 8:
            P.add(c, r, IT_ROLLBOULDER, (0, 0, 0, 0), "scenery", protect=False)
            rb_placed.append((r, c))

    # ---------------- Crystal Bowl (~24 Crystals parm0=0..2 in Zone 3) ----------------
    bowl_cells = [(int(r), int(c)) for r, c in np.argwhere(world["crystal_bowl"] & dry_mask) if (r, c) in bfs]
    P.scatter(24, bowl_cells, IT_CRYSTAL, lambda i: (i % 3, 0, 0, 0), "scenery", 2.0)

    # ---------------- Bushes (parm3=1 Tricer ambushes) ----------------
    # Zone 2 volcanic bushes & Zone 3 canyon bushes
    P.scatter(5, zone_flat[ZONE_LAVA], IT_BUSH, lambda i: (0, 0, 0, 1 if i < 4 else 0), "scenery", 4.0)
    P.scatter(5, zone_flat[ZONE_CANYON], IT_BUSH, lambda i: (0, 0, 0, 1 if i < 3 else 0), "scenery", 4.0)

    # ---------------- PowerUps (1/4-scale of original 86 -> ~24 normal, ~76 Pro) ----------------
    # Original: 86 across 25,420 tiles; Pro: 287 across 25,420 tiles (~3.34x)
    pow_plan = [
        (ZONE_MARSH, [POW_HEALTH, POW_LASER, POW_HEAT, POW_TRI, POW_SHIELD]),
        (ZONE_LAVA, [POW_HEALTH, POW_SHIELD, POW_LASER, POW_HEAT, POW_TRI, POW_SONIC]),
        (ZONE_CANYON, [POW_HEALTH, POW_LASER, POW_TRI, POW_SONIC, POW_HEAT, POW_NUKE]),
        (ZONE_LAGOON, [POW_HEALTH, POW_SHIELD, POW_LASER, POW_TRI, POW_HEAT, POW_SONIC, POW_NUKE]),
    ]
    for zi, plist in pow_plan:
        lst = list(plist)
        rng.shuffle(lst)
        got = P.scatter(len(lst), zone_dry[zi], IT_POWERUP, lambda i, l=lst: (l[i], 0, 0, 0), "powerup", 5.5)
        if got < len(lst):
            raise SystemExit(f"powerup placement failed in zone {zi} ({got}/{len(lst)})")

    # Pro extra powerups (~52 more -> ~76 total, matching Level1Pro's ~3.3x ratio)
    pro_pows = (
        [POW_TRI] * 14 + [POW_LASER] * 14 + [POW_NUKE] * 8
        + [POW_HEAT] * 7 + [POW_HEALTH] * 5 + [POW_SHIELD] * 4
    )
    rng.shuffle(pro_pows)
    for i, pw in enumerate(pro_pows):
        zi = i % 4
        P.scatter(1, zone_dry[zi], IT_POWERUP, lambda _n, p=pw: (p, 0, 0, 0), "powerup", 5.0, pro_only=True)

    # ---------------- Enemies (1/4-scale of original 571 -> ~145 normal; Pro ~320) ----------------
    enemy_plan = [
        # Zone 1 Emerald Marsh: Rex + Stego (+ light Spitter)
        (ZONE_MARSH, [(IT_REX, 10), (IT_STEGO, 12)]),
        # Zone 2 Obsidian Lava Gorge: Ptera + Tricer + Rex
        (ZONE_LAVA, [(IT_PTERA, 12), (IT_TRICER, 8), (IT_REX, 14)]),
        # Zone 3 Crystal Canyon & Oasis: Spitter + Rex + Tricer + Ptera
        (ZONE_CANYON, [(IT_SPITTER, 14), (IT_REX, 12), (IT_TRICER, 7), (IT_PTERA, 7)]),
        # Zone 4 Ancient Nest Lagoon: mixed endgame
        (ZONE_LAGOON, [(IT_REX, 16), (IT_SPITTER, 14), (IT_PTERA, 10), (IT_STEGO, 8), (IT_TRICER, 6)]),
    ]
    pro_mult = {
        IT_TRICER: 108 / 62,
        IT_REX: 403 / 188,
        IT_PTERA: 175 / 87,
        IT_STEGO: 146 / 79,
        IT_SPITTER: 430 / 155,
    }
    for zi, plan in enemy_plan:
        for et, n in plan:
            got = P.scatter(n, zone_dry[zi], et, lambda _n: (0, 0, 0, 0), "enemy", 2.0)
            if got < n:
                raise SystemExit(f"enemy placement failed zone={zi} type={et} ({got}/{n})")
            pro_extra = round(n * (pro_mult[et] - 1.0))
            P.scatter(pro_extra, zone_dry[zi], et, lambda _n: (0, 0, 0, 0), "enemy", 2.0, pro_only=True)

    # ---------------- Scenery (Trees, Mushrooms, Boulders, SporePods) ----------------
    # Zone 1 Marsh: Fern/Bamboo/Cypress/StickPalm groves, Mushrooms, Boulders, SporePods
    P.scatter(42, zone_flat[ZONE_MARSH], IT_TREE,
              lambda i: (rng.choice((TREE_FERN, TREE_BAMBOO, TREE_CYPRESS, TREE_STICKPALM)), 0, 0, 0),
              "tree", 2.5)
    P.scatter(10, zone_flat[ZONE_MARSH], IT_MUSHROOM, lambda _n: (0, 0, 0, 0), "scenery", 2.8)
    P.scatter(5, zone_flat[ZONE_MARSH], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)
    P.scatter(8, zone_flat[ZONE_MARSH], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)

    # Zone 2 Lava Gorge: Fern/PinePalm, Boulders
    P.scatter(18, zone_flat[ZONE_LAVA], IT_TREE,
              lambda i: (rng.choice((TREE_FERN, TREE_PINEPALM)), 0, 0, 0),
              "tree", 3.0)
    P.scatter(6, zone_flat[ZONE_LAVA], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)

    # Zone 3 Crystal Canyon & Oasis: StickPalm/MainPalm/Fern, SporePods, Boulders
    oasis_shore = [
        (int(r), int(c)) for r, c in np.argwhere(
            dilate8(dilate8(is_water_bed)) & (zone == ZONE_CANYON) & (kind == KIND_FLOOR)
        ) if (r, c) in bfs
    ]
    P.scatter(14, oasis_shore, IT_TREE,
              lambda i: (rng.choice((TREE_MAINPALM, TREE_STICKPALM)), 0, 0, 0), "tree", 2.5)
    P.scatter(22, zone_flat[ZONE_CANYON], IT_TREE,
              lambda i: (rng.choice((TREE_STICKPALM, TREE_MAINPALM, TREE_FERN)), 0, 0, 0),
              "tree", 2.8)
    P.scatter(12, zone_flat[ZONE_CANYON], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 2.8)
    P.scatter(5, zone_flat[ZONE_CANYON], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)

    # Zone 4 Ancient Nest Lagoon: Bamboo/MainPalm/StickPalm/Fern, SporePods
    P.scatter(34, zone_flat[ZONE_LAGOON], IT_TREE,
              lambda i: (rng.choice((TREE_BAMBOO, TREE_MAINPALM, TREE_STICKPALM, TREE_FERN)), 0, 0, 0),
              "tree", 2.6)
    P.scatter(8, zone_flat[ZONE_LAGOON], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)

    return P, bfs, (sc, sr), bridge_points


# =========================================================================
# Serialization
# =========================================================================
def pack_items(items):
    blob = struct.pack(">i", len(items))
    for x, z, t, p, _ in items:
        blob += struct.pack(">hhh4bHii", x, z, t, p[0], p[1], p[2], p[3], 0, 0, 0)
    return blob


def write_ter(path, tex, hm_grid, path_layer, items, tile_bank, orig):
    used = sorted({int(v) & 0x0FFF for v in np.unique(hm_grid)})
    assert len(used) <= 300, f"too many hm tiles: {len(used)} > 300"
    remap = {o: n for n, o in enumerate(used)}
    lut = np.zeros(4096, np.uint16)
    for o, n in remap.items():
        lut[o] = n
    hm_out = (lut[hm_grid & 0x0FFF] | (hm_grid & 0xC000)).astype(">u2")
    hm_blob = b"".join(tile_bank[o].tobytes() for o in used)

    n = W * D
    tex_off = 40
    hm_off = tex_off + n * 2
    path_off = hm_off + n * 2
    obj_off = path_off + n * 2
    item_blob = pack_items(items)
    hmtile_off = obj_off + len(item_blob)
    attrib_off = hmtile_off + len(hm_blob)
    anim_off = attrib_off + len(orig.attrib_blob)
    header = struct.pack(">7i2h2i", tex_off, hm_off, path_off, obj_off, 0,
                         hmtile_off, 0, W, D, attrib_off, anim_off)
    out = b"".join((
        header,
        tex.astype(">u2").tobytes(),
        hm_out.tobytes(),
        path_layer.astype(">u2").tobytes(),
        item_blob,
        hm_blob,
        orig.attrib_blob,
        orig.anim_blob,
    ))
    path.write_bytes(out)
    return len(used)


def write_map_tga(path, tex, orig):
    rgb = orig.tile_avg[np.asarray(tex) & 0x0FFF]
    img = Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    idx = np.asarray(img, np.uint8)
    ncol = int(idx.max()) + 1
    pal = img.getpalette()[:ncol * 3]
    header = struct.pack("<BBBHHBHHHHBB",
                         0, 1, 9, 0, ncol, 24, 0, D, W, D, 8, 0x20)
    palette = bytearray()
    for i in range(ncol):
        r, g, b = pal[i * 3:i * 3 + 3]
        palette += bytes((b, g, r))
    body = bytearray()
    for row in idx:
        i = 0
        while i < W:
            run = 1
            while i + run < W and row[i + run] == row[i] and run < 128:
                run += 1
            if run >= 2:
                body += bytes((0x80 | (run - 1), row[i]))
                i += run
            else:
                j = i + 1
                while j < W and (j - i) < 128 and (j + 1 >= W or row[j] != row[j + 1]):
                    j += 1
                body += bytes((j - i - 1,)) + row[i:j].tobytes()
                i = j
    path.write_bytes(header + palette + bytes(body))


def write_preview(path, tex, mean_map, items, orig):
    S = 4
    rgb = orig.tile_avg[np.asarray(tex) & 0x0FFF].astype(np.float32)
    shade = 0.55 + 0.45 * (1.0 - np.abs(mean_map - 85.0) / 170.0)
    shade = np.clip(shade, 0.4, 1.0)
    rgb = (rgb * shade[..., None]).astype(np.uint8)
    big = np.kron(rgb, np.ones((S, S, 1), np.uint8))
    img = Image.fromarray(big, "RGB")
    dr = ImageDraw.Draw(img)
    egg_colors = ["#00ff40", "#ffa000", "#ff2020", "#00e0ff", "#ff00ff"]
    enemy_colors = {IT_REX: "#c00000", IT_TRICER: "#a05820", IT_PTERA: "#c060ff",
                    IT_STEGO: "#90a800", IT_SPITTER: "#3050ff"}
    for x, z, t, p, pro in items:
        if pro:
            continue
        cx, cy = x * S // TILE_PX, z * S // TILE_PX
        if t == IT_EGG:
            col = egg_colors[p[0]]
            dr.ellipse((cx - 6, cy - 6, cx + 6, cy + 6), outline=col, width=2)
            dr.text((cx - 3, cy - 6), "E", fill=col)
        elif t == IT_PORTAL:
            dr.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline="white", width=2)
            dr.text((cx - 3, cy - 6), "P", fill="white")
        elif t == IT_START:
            dr.ellipse((cx - 8, cy - 8, cx + 8, cy + 8), outline="yellow", width=3)
            dr.text((cx - 3, cy - 6), "S", fill="yellow")
        elif t == IT_STEPSTONE:
            dr.rectangle((cx - 3, cy - 3, cx + 3, cy + 3), fill="#d0d0d0", outline="black")
        elif t == IT_GASVENT:
            dr.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="#ffffff", outline="#60e0ff")
        elif t == IT_ROLLBOULDER:
            dr.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="#907050", outline="black")
        elif t in enemy_colors:
            dr.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill=enemy_colors[t])
        elif t == IT_POWERUP:
            dr.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="#ffff00", outline="black")
        elif t == IT_CRYSTAL:
            dr.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), fill="#80ffff")
        elif t == IT_WATER:
            dr.rectangle((cx - 2, cy - 2, cx + 2, cy + 2), outline="#4080ff")
        elif t == IT_LAVA:
            dr.rectangle((cx - 2, cy - 2, cx + 2, cy + 2), outline="#ff8000")

    legend = Image.new("RGB", (img.width, 72), (16, 16, 16))
    ld = ImageDraw.Draw(legend)
    ld.text((8, 6), "S start   P portal   E egg (color=species 0..4)   gray-box=StepStone   white-dot=GasVent", fill="white")
    ld.text((8, 24), "dots: red=rex brown=tricer purple=ptera olive=stego blue=spitter yellow=powerup cyan=crystal", fill="white")
    ld.text((8, 42), "squares: blue=water patch orange=lava patch", fill="white")
    out = Image.new("RGB", (img.width, img.height + 72))
    out.paste(img, (0, 0))
    out.paste(legend, (0, img.height))
    out.save(path)


# =========================================================================
# Strict Self-Validation
# =========================================================================
def validate_ter(path, orig, label):
    checks = []
    data = path.read_bytes()
    h = struct.unpack_from(">7i2h2i", data, 0)
    tex_off, hm_off, path_off, obj_off, _, hmtile_off, _, w, d, attrib_off, anim_off = h
    ok = (w, d) == (W, D) and tex_off == 40
    n = w * d
    tex = np.frombuffer(data, ">u2", n, tex_off).reshape(d, w)
    hm = np.frombuffer(data, ">u2", n, hm_off).reshape(d, w)
    pathl = np.frombuffer(data, ">u2", n, path_off).reshape(d, w)
    n_hm = (attrib_off - hmtile_off) // 1024
    hm_tiles = np.frombuffer(data, "u1", n_hm * 1024, hmtile_off).reshape(n_hm, 32, 32)
    attribs = [struct.unpack_from(">Hhbbh", data, attrib_off + i * 8)
               for i in range((anim_off - attrib_off) // 8)]
    num_items = struct.unpack_from(">i", data, obj_off)[0]
    items = [struct.unpack_from(">hhh4bHii", data, obj_off + 4 + i * 20)
             for i in range(num_items)]

    checks.append(("1 header/sizes", ok and n_hm <= 300 and int((tex & 0xFFF).max()) < 258,
                   f"{w}x{d}, {n_hm} hm tiles (<=300), max tex {(tex & 0xFFF).max()}, {num_items} items"))

    # Item rules
    xs = [it[0] for it in items]
    sorted_ok = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    starts = [it for it in items if it[2] == IT_START]
    eggs = [it for it in items if it[2] == IT_EGG]
    egg_by_sp = Counter(e[3] for e in eggs)
    eggs_ok = all(egg_by_sp.get(s, 0) == 5 for s in range(5)) and all((e[6] & 1) == 1 for e in eggs)
    portals = sorted(it[3] for it in items if it[2] == IT_PORTAL)
    rollboulder_count = sum(1 for it in items if it[2] == IT_ROLLBOULDER)
    types_ok = all(0 <= it[2] <= 19 for it in items)
    border_ok = all(4 <= it[0] // 32 < W - 4 and 4 <= it[1] // 32 < D - 4 for it in items)
    checks.append(("2 item rules", sorted_ok and len(starts) == 1 and eggs_ok
                   and portals == [0, 1, 2, 3] and rollboulder_count >= 4 and types_ok and border_ok,
                   f"sorted={sorted_ok} starts={len(starts)} eggs/sp={dict(egg_by_sp)} "
                   f"portals={portals} rollboulders={rollboulder_count}"))

    # Reconstruct exact vertex heightfield V_read[r,c] = corner[0,0] post-flip (just like Terrain.c!)
    V_read = np.zeros((d, w), dtype=np.int32)
    tile_min = np.zeros((d, w), dtype=np.int32)
    tile_max = np.zeros((d, w), dtype=np.int32)
    for r in range(d):
        for c in range(w):
            v = int(hm[r, c])
            t = hm_tiles[v & 0x0FFF]
            if v & 0x8000:
                t = t[:, ::-1]
            if v & 0x4000:
                t = t[::-1, :]
            V_read[r, c] = int(t[0, 0])
            tile_min[r, c] = int(t.min())
            tile_max[r, c] = int(t.max())

    # Check 3: Containment & BFS
    border = np.zeros((d, w), bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    cont_ok = bool(np.all(pathl[border] == PATH_SOLID_ALL) and np.all(V_read[border] == 255))
    walk_open = (pathl == PATH_OPEN) & (V_read <= 185)
    st = starts[0]
    sr, sc = st[1] // 32, st[0] // 32
    seen = np.zeros((d, w), bool)
    dq = deque([(sr, sc)])
    seen[sr, sc] = True
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < d and 0 <= nc < w and not seen[nr, nc] and walk_open[nr, nc]:
                seen[nr, nc] = True
                dq.append((nr, nc))
    edge_touch = bool(np.any(walk_open[0, :]) or np.any(walk_open[-1, :]) or
                      np.any(walk_open[:, 0]) or np.any(walk_open[:, -1]))
    bfs_count = int(seen.sum())
    checks.append(("3 containment/BFS", cont_ok and not edge_touch and 5800 <= bfs_count <= 7000,
                   f"border_solid={cont_ok}, edge_touch={edge_touch}, bfs_tiles={bfs_count}"))

    # Check 4: Connectivity of all 25 eggs, 4 portals, all powerups, crystals, mushrooms, enemies
    check_types = {IT_EGG, IT_PORTAL, IT_POWERUP, IT_CRYSTAL, IT_MUSHROOM} | set(ENEMY_TYPES)
    unreachable = [(it[2], it[0] // 32, it[1] // 32)
                   for it in items if it[2] in check_types and not seen[it[1] // 32, it[0] // 32]]
    checks.append(("4 connectivity", not unreachable,
                   f"unreachable={unreachable[:6]} (25 eggs + 4 portals + all powerups/enemies reachable)"))

    # Check 5: Exact 8x8 Water & Lava coverage [cr-3..cr+4, cc-3..cc+4] + ZERO dry TILE_ATTRIB_WATER / TILE_ATTRIB_LAVA leaks
    flags_g = np.array([attribs[int(t) & 0xFFF][0] for t in tex.ravel()]).reshape(d, w)
    w_attr = (flags_g & TILE_ATTRIB_WATER) != 0
    l_attr = (flags_g & TILE_ATTRIB_LAVA) != 0

    dry_water_leaks = int(np.sum(w_attr & (tile_max > 47)))
    dry_lava_leaks = int(np.sum(l_attr & ((V_read != 67) | (tile_max != 67))))

    w_cov = np.zeros((d, w), bool)
    for it in items:
        if it[2] == IT_WATER:
            cc, cr = it[0] // 32, it[1] // 32
            c0, c1 = max(0, cc - 3), min(w, cc + 5)
            r0, r1 = max(0, cr - 3), min(d, cr + 5)
            w_cov[r0:r1, c0:c1] = True
    hm41_mask = (tile_max == 41) & (tile_min == 41) & (pathl == PATH_OPEN)
    uncov_water = int(np.sum((hm41_mask | w_attr) & ~w_cov))

    l_cov = np.zeros((d, w), bool)
    for it in items:
        if it[2] == IT_LAVA:
            cc, cr = it[0] // 32, it[1] // 32
            c0, c1 = max(0, cc - 3), min(w, cc + 5)
            r0, r1 = max(0, cr - 3), min(d, cr + 5)
            l_cov[r0:r1, c0:c1] = True
    uncov_lava = int(np.sum(l_attr & ~l_cov))

    checks.append(("5 water/lava",
                   dry_water_leaks == 0 and dry_lava_leaks == 0 and uncov_water == 0 and uncov_lava == 0,
                   f"uncov_water={uncov_water}, uncov_lava={uncov_lava}, "
                   f"dry_water_leaks={dry_water_leaks}, dry_lava_leaks={dry_lava_leaks}"))

    # Check 6: No walkable vertex spikes + no 182 on path=0 + smooth walkable steps + perimeter corner check
    has_182_walk = int(np.sum(((hm & 0xFFF) == 182) & (pathl == PATH_OPEN)))
    walk_p0 = (pathl == PATH_OPEN)
    dx_walk = np.abs(V_read[:, 1:] - V_read[:, :-1])
    dz_walk = np.abs(V_read[1:, :] - V_read[:-1, :])
    max_step_ew = int(dx_walk[walk_p0[:, :-1] & walk_p0[:, 1:]].max())
    max_step_ns = int(dz_walk[walk_p0[:-1, :] & walk_p0[1:, :]].max())
    max_step_all = max(max_step_ew, max_step_ns)
    walk_nonspire = walk_p0 & (V_read <= 128)
    max_step_nonspire = max(
        int(dx_walk[walk_nonspire[:, :-1] & walk_nonspire[:, 1:]].max()),
        int(dz_walk[walk_nonspire[:-1, :] & walk_nonspire[1:, :]].max()),
    )
    # Check pixel[0,0] continuity against tile corners [0,31], [31,0], [31,31] on path=0 AND perimeter corners!
    corner_mismatch = 0
    perimeter_spikes = 0
    for r in range(d - 1):
        for c in range(w - 1):
            v = int(hm[r, c])
            t = hm_tiles[v & 0x0FFF]
            if v & 0x8000:
                t = t[:, ::-1]
            if v & 0x4000:
                t = t[::-1, :]
            if walk_p0[r, c]:
                if (int(t[0, 31]) != int(V_read[r, c + 1])
                        or int(t[31, 0]) != int(V_read[r + 1, c])
                        or int(t[31, 31]) != int(V_read[r + 1, c + 1])):
                    corner_mismatch += 1
                # Any corner of a walk tile coming from a non-walk tile must be 85 (no cliff spikes!)
                for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    if not walk_p0[r + dr, c + dc] and int(V_read[r + dr, c + dc]) != 85:
                        perimeter_spikes += 1
            elif walk_p0[r, c + 1] or walk_p0[r + 1, c] or walk_p0[r + 1, c + 1] or walk_p0[r - 1, c] or walk_p0[r, c - 1]:
                # Non-walk tile bordering walk: corner[0,0] must match V_read[r,c], and if sharing a vertex with walk, no spike
                if walk_p0[r - 1, c - 1] or walk_p0[r - 1, c] or walk_p0[r, c - 1]:
                    if int(V_read[r, c]) != 85:
                        perimeter_spikes += 1

    checks.append(("6 vertex continuity",
                   has_182_walk == 0 and corner_mismatch == 0 and perimeter_spikes == 0
                   and max_step_nonspire <= 20 and max_step_all <= 22,
                   f"182_on_path0={has_182_walk}, corner_mismatch={corner_mismatch}, "
                   f"perimeter_spikes={perimeter_spikes}, max_step_nonspire={max_step_nonspire}, "
                   f"max_step_all={max_step_all}"))

    # Check 7: StepStone spacing <= 3.5 tiles & on flat hm=67 lava (each bridge >= 3 stones)
    stones = [(it[0] / 32.0, it[1] / 32.0, it[6]) for it in items if it[2] == IT_STEPSTONE]
    stone_ok = len(stones) >= 6 and all(s[2] == 1 for s in stones)
    max_stone_nn = 0.0
    for i, (cx, cz, _) in enumerate(stones):
        r, c = int(cz), int(cx)
        if not (tile_min[r, c] == 67 and tile_max[r, c] == 67 and l_attr[r, c]):
            stone_ok = False
        nn = min(math.hypot(cx - ox, cz - oz) for j, (ox, oz, _) in enumerate(stones) if j != i)
        max_stone_nn = max(max_stone_nn, nn)
    if max_stone_nn > 3.55:
        stone_ok = False
    checks.append(("7 stepstones", stone_ok,
                   f"count={len(stones)}, all_reincarnate={all(s[2]==1 for s in stones)}, max_nn={max_stone_nn:.2f} tiles"))

    # Check 8: Spacing (no duplicate non-hazard tiles; enemies/powerups separated)
    tiles_used = Counter((it[0] // 32, it[1] // 32) for it in items)
    dup = [k for k, v in tiles_used.items() if v > 1]
    checks.append(("8 item spacing", not dup, f"dup_tiles={len(dup)}"))

    floor_pct = 100.0 * np.count_nonzero(walk_open) / (w * d)
    cliff_pct = 100.0 * np.count_nonzero(V_read == 255) / (w * d)
    stats = dict(floor_pct=floor_pct, cliff_pct=cliff_pct, items=num_items,
                 hm_tiles=n_hm, bfs_cells=bfs_count)
    return checks, stats, items


def validate_tga(path):
    b = path.read_bytes()
    (idlen, cmtype, imtype, cmorig, cmlen, cmdepth,
     xo, yo, w, h, bpp, desc) = struct.unpack_from("<BBBHHBHHHHBB", b, 0)
    ok = bool(idlen == 0 and cmtype == 1 and imtype == 9 and cmorig == 0
              and cmlen <= 256 and cmdepth == 24 and (w, h) == (W, D)
              and bpp == 8 and (desc & 0x20))
    return ok, (f"type={imtype} (RLE colormapped), {w}x{h}, {bpp}bpp, "
                f"{cmlen} colors, descriptor=0x{desc:02x} (top-left origin)")


# =========================================================================
def main():
    import random
    rng = random.Random(SEED)
    orig = Original()

    print("building world ...")
    world = build_world(rng)

    print("synthesizing exact-corner heightmap tiles ...")
    hm_grid, path_layer, tile_bank = build_heightmaps(world, orig)

    print("painting biome textures ...")
    tex = paint_textures(world, hm_grid, orig, rng)

    print("placing items ...")
    P, bfs, start, bridge_points = build_items(rng, world, hm_grid, path_layer)

    items_sorted = sorted(P.items, key=lambda it: (it[0], it[1]))
    normal_items = [it for it in items_sorted if not it[4]]
    pro_items = items_sorted

    OUT_TERRAIN.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    write_ter(OUT_TERRAIN / "Level1.ter", tex, hm_grid, path_layer, normal_items, tile_bank, orig)
    write_ter(OUT_TERRAIN / "Level1Pro.ter", tex, hm_grid, path_layer, pro_items, tile_bank, orig)
    shutil.copyfile(ORIG_DIR / "Level1.trt", OUT_TERRAIN / "Level1.trt")
    write_map_tga(OUT_IMAGES / "Map.tga", tex, orig)
    write_preview(PREVIEW_PATH, tex, world["tmean"], P.items, orig)

    print("\n=== VALIDATION REPORT ===")
    all_ok = True
    for label, f in (("Level1.ter", OUT_TERRAIN / "Level1.ter"),
                     ("Level1Pro.ter", OUT_TERRAIN / "Level1Pro.ter")):
        checks, st, items = validate_ter(f, orig, label)
        print(f"\n--- {label} ---")
        for name, ok, detail in checks:
            all_ok &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        print(f"  stats: floor {st['floor_pct']:.1f}%  cliff255 {st['cliff_pct']:.1f}%  "
              f"items {st['items']}  hm tiles {st['hm_tiles']}  BFS cells {st['bfs_cells']}")
        tc = Counter(it[2] for it in items)
        print(f"  item counts by type: {dict(sorted(tc.items()))}")

    ok8, det8 = validate_tga(OUT_IMAGES / "Map.tga")
    all_ok &= ok8
    print(f"\n  [{'PASS' if ok8 else 'FAIL'}] 9 Map.tga format: {det8}")

    zone_names = ["Emerald Marsh", "Obsidian Lava Gorge", "Crystal Canyon", "Ancient Nest Lagoon"]
    zone = world["zone"]
    print("\n  zone breakdown (BFS tiles & normal enemies):")
    for zi, zname in enumerate(zone_names):
        ztiles = sum(1 for r, c in bfs if zone[r, c] == zi)
        zen = sum(1 for x, z, t, p, pro in P.items
                  if not pro and t in ENEMY_TYPES and zone[z // 32, x // 32] == zi)
        print(f"    {zname:22s}: {ztiles:4d} BFS tiles, {zen:3d} enemies "
              f"({1000 * zen / max(1, ztiles):.1f}/1k)")

    if not all_ok:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")

    if BUNDLE.is_dir():
        (BUNDLE / "Terrain").mkdir(parents=True, exist_ok=True)
        (BUNDLE / "Images").mkdir(parents=True, exist_ok=True)
        for f in ("Level1.ter", "Level1Pro.ter", "Level1.trt"):
            shutil.copyfile(OUT_TERRAIN / f, BUNDLE / "Terrain" / f)
        shutil.copyfile(OUT_IMAGES / "Map.tga", BUNDLE / "Images" / "Map.tga")
        print(f"installed into {BUNDLE}")


if __name__ == "__main__":
    main()
