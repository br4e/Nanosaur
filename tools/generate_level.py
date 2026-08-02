#!/usr/bin/env python3
"""Canonical generator for the "Caldera Loop" custom Level 1 for Nanosaur.

Produces:
  Data/Terrain/Level1.ter, Data/Terrain/Level1Pro.ter (same geometry, Pro has
  ~30% more enemies/powerups), Data/Terrain/Level1.trt (verbatim copy of the
  original tileset), Data/Images/Map.tga (GPS map, RLE colormapped 8-bit,
  matching the format the game's TGA.c loader expects), and
  tools/level_preview.png (human-review render).

Design: a ring of five biome basins arranged counterclockwise around a large
impassable central mesa, joined by cliff-walled connector canyons.

All heightmap geometry reuses the original game's heightmap tiles.  Ramp/cliff
transitions are picked by edge/corner matching against the desired corner
heights so that adjacent tiles line up without cracks.  Cliff-band textures
are copied from the original level's (heightmap-tile, flip) -> texture
association so walls look authentic.

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
SEED = 20260801

# ---------------------------------------------------------------- item types
IT_START, IT_POWERUP, IT_TRICER, IT_REX, IT_LAVA = 0, 1, 2, 3, 4
IT_EGG, IT_GASVENT, IT_PTERA, IT_STEGO, IT_PORTAL = 5, 6, 7, 8, 9
IT_TREE, IT_BOULDER, IT_MUSHROOM, IT_BUSH, IT_WATER = 10, 11, 12, 13, 14
IT_CRYSTAL, IT_SPITTER, IT_STEPSTONE, IT_ROLLBOULDER, IT_SPOREPOD = 15, 16, 17, 18, 19
ENEMY_TYPES = (IT_TRICER, IT_REX, IT_PTERA, IT_STEGO, IT_SPITTER)

# powerup parm0 (verified against AddPowerUp in src/Items/Triggers.c)
POW_HEAT, POW_LASER, POW_TRI, POW_HEALTH, POW_SHIELD, POW_NUKE, POW_SONIC = range(7)

# ------------------------------------------------------------- terrain kinds
CLIFF, CRAMP_L, CRAMP_U = 0, 1, 2
FLOOR, LOW67, PAD109 = 3, 4, 5
LK_SOFT, LK_SHELF, LK_DEEP, LK_BED = 6, 7, 8, 9
LV_SOFT, LV_BED = 10, 11
MD_RAMP, MD_PAD = 12, 13
N_KINDS = 14

# (lo, hi) plateau range each kind spans (heightmap pixel values, worldY = px*4)
KIND_RANGE = {
    CLIFF: (255, 255), CRAMP_L: (85, 149), CRAMP_U: (151, 254),
    FLOOR: (85, 85), LOW67: (67, 67), PAD109: (109, 109),
    LK_SOFT: (67, 85), LK_SHELF: (67, 67), LK_DEEP: (41, 67), LK_BED: (41, 41),
    LV_SOFT: (67, 85), LV_BED: (67, 67),
    MD_RAMP: (85, 224), MD_PAD: (224, 224),
}
FLAT_TILE = {CLIFF: 0, FLOOR: 6, LOW67: 7, PAD109: 5,
             LK_SHELF: 7, LK_BED: 8, LV_BED: 7, MD_PAD: 31}
# Lake/lava/mesa ramps use SSE; cliff walls use the original fixed kit (below).
RAMP_CANDIDATES = {
    LK_SOFT: (22, 37, 48, 38, 23),
    LV_SOFT: (22, 37, 48, 38, 23),
    LK_DEEP: (24, 40, 39, 51, 35),
    MD_RAMP: (1, 32, 2, 17),
}
# Original Level1 cliff kit: 2-tile orth stacks + 3-tile diagonal stamps.
# Facing = direction toward the low/walkable side.
CLIFF_ORTH = {  # facing -> (lower_tid, upper_tid, fx, fy)
    "E": (167, 166, 0, 0),
    "W": (167, 166, 1, 0),
    "S": (235, 217, 0, 0),
    "N": (235, 217, 0, 1),
}
# Diagonal high-tip -> flip; 180 stamps toward cliff, 182 toward floor.
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
CLIFF_PATH43 = {167, 235, 182}
KIT_HM_IDS = sorted(
    {t for v in RAMP_CANDIDATES.values() for t in v}
    | set(FLAT_TILE.values())
    | {166, 167, 180, 181, 182, 217, 235}
)

WALKABLE_KINDS = {FLOOR, LOW67, PAD109}          # where regular items may go
BOWL_KINDS = {LK_SOFT, LK_SHELF, LK_DEEP, LK_BED, LV_SOFT, LV_BED}

# ------------------------------------------------------------ biome geometry
# (name, center(col,row), radii(rx,rz)) — 40% smaller, ring pulled tighter
BIOMES = [
    ("Emerald Shallows", (101, 216), (17, 17)),
    ("Fern Canyons",     (97, 152), (14, 20)),
    ("Ember Flats",      (135, 124), (23, 16)),
    ("Crystal Scar",     (172, 162), (16, 19)),
    ("Nest Caldera",     (166, 222), (18, 17)),
]
MESA = ((135, 180), (22, 26))
# short connector bridges between nearby basin edges
CONNECTORS = [
    ("Shallows->Fern",  [(100, 200), (98, 186), (97, 172)], (0.4, 0.7)),
    ("Fern->Ember",     [(110, 145), (120, 135), (128, 125)], (0.45,)),
    ("Ember->Crystal",  [(150, 125), (158, 140), (165, 152)], (0.35, 0.7)),
    ("Crystal->Nest",   [(172, 175), (170, 190), (168, 205)], (0.5,)),
    ("Nest->Shallows",  [(150, 225), (130, 225), (112, 218)], (0.35, 0.7)),
]
FERN_BLOBS = [(90, 145, 3), (102, 155, 3), (92, 160, 4), (100, 165, 3)]

# Feature intents — positions are auto-snapped onto real floor after geometry.
# Lakes: (biome_idx, half_w, half_h) — rectangular deep beds so an 8x8 water
# item fits entirely in LK_BED (never on soft fringe / flat floor).
LAKE_INTENTS = [
    (0, 5, 5),   # 11x11 bed
    (3, 5, 4),
    (4, 5, 5),
]
# Ember lava: cover most of biome 2; sparse jump platforms farther apart.
EMBER_BIOME = 2
EMBER_PLATFORM_COUNT = 6          # including the start island
EMBER_PLATFORM_MIN_GAP = 14       # tiles between platform centers
EMBER_PLATFORM_RADIUS = 1         # 3x3 floor islands
# Nest pads: how many to try in Nest Caldera (biome 4)
NEST_PAD_COUNT = 2

# ------------------------------------------------------------ texture weaves
SOUTH_GRASS = [189, 190, 200, 201]
NORTH_GRASS_FLIP = [17, 36, 0x8011, 0x8024, 0x4011, 0xC011]
EMBER_MESA = [0x8011, 0x8024, 0x4011, 0xC011]
JUNGLE_MIX = [144, 145, 160, 161, 38, 39]
CRYSTAL_MIX = [144, 145, 160, 161]
NEST_WATERSIDE = [223, 224, 231, 232]
LAVA_BED = [137, 138, 151, 152, 0xC000 | 137, 0xC000 | 138, 0x8000 | 151, 0x4000 | 152]
LAVA_FRINGE = [164, 165, 148, 149]
WATER_BED = [23, 24, 42, 43]
SHORE = [74, 75, 78, 89, 90]
WALL_FALLBACK = [64, 65, 66, 122, 80, 81]

TILE_ATTRIB_LAVA = 1 << 5
TILE_ATTRIB_WATER = 1 << 7


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

    def tile_pixels(self, tid, fx, fy):
        t = self.hm_tiles[tid].astype(np.float64)
        if fx:
            t = t[:, ::-1]
        if fy:
            t = t[::-1, :]
        return t


def tile_stats(orig):
    """(tid,fx,fy) -> (corners NW,NE,SW,SE), (edge means N,S,W,E), mean."""
    stats = {}
    for tid in range(len(orig.hm_tiles)):
        for fx in (0, 1):
            for fy in (0, 1):
                t = orig.tile_pixels(tid, fx, fy)
                stats[(tid, fx, fy)] = (
                    (t[0, 0], t[0, -1], t[-1, 0], t[-1, -1]),
                    (t[0].mean(), t[-1].mean(), t[:, 0].mean(), t[:, -1].mean()),
                    t.mean(),
                )
    return stats


def derive_hm_to_tex(orig):
    """Most common texture value used on each (hm tile, flip) in the original."""
    counters = {}
    ids = orig.hm & 0x0FFF
    mask = np.isin(ids, KIT_HM_IDS)
    for r, c in zip(*np.nonzero(mask)):
        v = int(orig.hm[r, c])
        key = (v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1)
        counters.setdefault(key, Counter())[int(orig.tex[r, c])] += 1
    return {k: cnt.most_common(1)[0][0] for k, cnt in counters.items()}


# =========================================================================
# Geometry
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


def build_geometry(rng):
    cc, rr = np.meshgrid(np.arange(W), np.arange(D))

    def ellipse(cx, cz, rx, rz):
        return ((cc - cx) / rx) ** 2 + ((rr - cz) / rz) ** 2 <= 1.0

    walk = np.zeros((D, W), bool)
    for _, (cx, cz), (rx, rz) in BIOMES:
        walk |= ellipse(cx, cz, rx, rz)

    conn_masks = []
    for _, pts, pockets in CONNECTORS:
        m = np.zeros((D, W), bool)
        seg_len = [math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
        total = sum(seg_len)
        acc = 0.0
        for i in range(len(pts) - 1):
            (x0, z0), (x1, z1) = pts[i], pts[i + 1]
            steps = max(2, int(seg_len[i] * 2))
            for s in range(steps + 1):
                f = s / steps
                t = (acc + f * seg_len[i]) / total
                hw = 4.0
                for tp in pockets:
                    hw += 2.5 * math.exp(-((t - tp) / 0.07) ** 2)
                x, z = x0 + f * (x1 - x0), z0 + f * (z1 - z0)
                r0, r1 = int(z - hw), int(z + hw) + 1
                c0, c1 = int(x - hw), int(x + hw) + 1
                sub = (
                    (cc[r0:r1, c0:c1] - x) ** 2 + (rr[r0:r1, c0:c1] - z) ** 2
                ) <= hw * hw
                m[r0:r1, c0:c1] |= sub
            acc += seg_len[i]
        conn_masks.append(m)
        walk |= m

    # impassable central mesa + Fern canyon blobs
    (mx, mz), (mrx, mrz) = MESA
    walk &= ~ellipse(mx, mz, mrx, mrz)
    for bx, bz, br in FERN_BLOBS:
        walk &= ~ellipse(bx, bz, br, br)

    # containment margin: nothing walkable within 6 tiles of the map edge
    walk[:6, :] = walk[-6:, :] = False
    walk[:, :6] = walk[:, -6:] = False

    kind = np.full((D, W), CLIFF, np.uint8)
    kind[walk] = FLOOR

    # cliff ramp bands: chebyshev distance 1 and 2 from walkable
    d1 = dilate8(walk) & ~walk
    d2 = dilate8(dilate8(walk)) & ~walk & ~d1
    kind[d1] = CRAMP_L
    kind[d2] = CRAMP_U

    feature = np.zeros((D, W), bool)     # cells reserved by lakes/lava/pads

    def ring_ok(mask):
        return bool(mask.any() and np.all(kind[mask] == FLOOR) and not np.any(feature[mask]))

    def biome_floor_cells(bi, margin=3):
        """Interior floor cells of a biome, kept clear of walls."""
        bx, bz = BIOMES[bi][1]
        rx, rz = BIOMES[bi][2]
        interior = walk & ellipse(bx, bz, max(1, rx - margin), max(1, rz - margin))
        interior &= (kind == FLOOR) & ~feature
        # also keep a 1-tile buffer from non-floor
        interior &= ~dilate8(kind != FLOOR)
        return np.argwhere(interior)

    def core_fits_water(bed, deep):
        """True if some 8x8 window lies entirely in the deep basin (bed|deep)."""
        core = bed | deep
        rows, cols = np.where(core)
        if len(rows) == 0:
            return False
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        c0, c1 = int(cols.min()), int(cols.max()) + 1
        for r in range(r0, r1 - 7):
            for c in range(c0, c1 - 7):
                if np.all(core[r:r + 8, c:c + 8]):
                    return True
        return False

    def try_lake_at(cx, cz, hw, hh):
        # Rectangular deep bed — axis-aligned so an 8x8 water item can sit inside.
        bed = ((np.abs(cc - cx) <= hw) & (np.abs(rr - cz) <= hh))
        deep = dilate8(bed) & ~bed
        shelf = dilate8(bed | deep) & ~(bed | deep)
        soft = dilate8(bed | deep | shelf) & ~(bed | deep | shelf)
        # Soft is optional edge paint; only the carved basin must be clear floor.
        if not ring_ok(bed | deep | shelf):
            return None
        soft = soft & (kind == FLOOR) & ~feature
        if not core_fits_water(bed, deep):
            return None
        return bed, deep, shelf, soft, (bed | deep | shelf | soft)

    def place_lake(bi, hw, hh):
        """Search biome floor for a recessed lake footprint; prefer near center."""
        nonlocal feature
        bx, bz = BIOMES[bi][1]
        cells = biome_floor_cells(bi, margin=max(hw, hh) + 3)
        if len(cells) == 0:
            return None
        # Sort by distance to biome center so lakes prefer the basin middle.
        order = sorted(range(len(cells)),
                       key=lambda i: (cells[i][1] - bx) ** 2 + (cells[i][0] - bz) ** 2)
        trials = order[:40] + order[40::max(1, len(order) // 30)]
        rng.shuffle(trials[5:])
        for i in trials:
            r, c = int(cells[i][0]), int(cells[i][1])
            got = try_lake_at(c, r, hw, hh)
            if got is None and (hw > 4 or hh > 4):
                got = try_lake_at(c, r, max(4, hw - 1), max(4, hh - 1))
            if got is None:
                continue
            bed, deep, shelf, soft, whole = got
            kind[bed], kind[deep], kind[shelf], kind[soft] = (
                LK_BED, LK_DEEP, LK_SHELF, LK_SOFT)
            feature |= dilate8(dilate8(whole))
            return bed
        return None

    lake_cells = []
    for bi, rx, rz in LAKE_INTENTS:
        bed = place_lake(bi, rx, rz)
        if bed is not None:
            lake_cells.append((bi, bed))
        else:
            print(f"  warn: could not place lake intent biome={bi} size=({rx},{rz})")

    # ---- Ember Flats: lava covers most of the biome; sparse jump platforms ----
    lava_cells = []
    ember_platforms = []  # (r, c) centers of floor islands
    bi = EMBER_BIOME
    bx, bz = BIOMES[bi][1]
    ember_mask = walk & ellipse(bx, bz, BIOMES[bi][2][0], BIOMES[bi][2][1])
    ember_floor = ember_mask & (kind == FLOOR) & ~feature

    cand = np.argwhere(ember_floor)
    rng.shuffle(cand)

    def far_enough(cr, cc0, existing, gap=EMBER_PLATFORM_MIN_GAP):
        for er, ec in existing:
            if (cr - er) ** 2 + (cc0 - ec) ** 2 < gap * gap:
                return False
        return True

    # Prefer start-friendly south/east island first, then spread.
    seed_plats = [
        (bz + BIOMES[bi][2][1] // 3, bx + BIOMES[bi][2][0] // 4),
        (bz, bx), (bz - 4, bx - 6), (bz - 4, bx + 6),
        (bz + 4, bx), (bz, bx - 8), (bz, bx + 8),
    ]
    for cr, cc0 in seed_plats + [tuple(x) for x in cand]:
        if len(ember_platforms) >= EMBER_PLATFORM_COUNT or len(cand) == 0:
            break
        cr, cc0 = int(cr), int(cc0)
        d2 = (cand[:, 0] - cr) ** 2 + (cand[:, 1] - cc0) ** 2
        i = int(np.argmin(d2))
        pr, pc = int(cand[i][0]), int(cand[i][1])
        foot = ((rr - pr) ** 2 + (cc - pc) ** 2) <= EMBER_PLATFORM_RADIUS ** 2
        if not (np.all(kind[foot] == FLOOR) and not np.any(feature[foot])):
            continue
        if not far_enough(pr, pc, ember_platforms):
            continue
        ember_platforms.append((pr, pc))

    plat_union = np.zeros((D, W), bool)
    for pr, pc in ember_platforms:
        foot = ((rr - pr) ** 2 + (cc - pc) ** 2) <= EMBER_PLATFORM_RADIUS ** 2
        plat_union |= foot

    # Flood remaining ember floor with recessed lava (core bed + soft fringe).
    lava_area = ember_floor & ~plat_union
    if lava_area.any():
        core = lava_area & ~dilate8(~lava_area)
        if int(core.sum()) < 8:
            core = lava_area
        soft = dilate8(core) & ~core & (kind == FLOOR) & ~plat_union
        leftover = lava_area & ~core & ~soft & (kind == FLOOR)
        kind[core] = LV_BED
        kind[soft | leftover] = LV_SOFT
        feature |= dilate8(core | soft | leftover | plat_union)
        lava_cells.append((core, True))
    print(f"  ember platforms={len(ember_platforms)} lava_core="
          f"{int(lava_cells[0][0].sum()) if lava_cells else 0}")

    # ---- Nest pads: auto-snap into Nest Caldera floor ----
    pad_cells = []
    nest_bi = 4
    nest_cands = biome_floor_cells(nest_bi, margin=4)
    rng.shuffle(nest_cands)
    for r, c in nest_cands:
        if len(pad_cells) >= NEST_PAD_COUNT:
            break
        r, c = int(r), int(c)
        # 2x2 pad with 1-tile ramp ring → block [r-1:r+3, c-1:c+3]
        if not (1 <= r < D - 3 and 1 <= c < W - 3):
            continue
        block = np.zeros((D, W), bool)
        block[r - 1:r + 3, c - 1:c + 3] = True
        if not ring_ok(block):
            continue
        # keep pads apart
        if any((r - pr) ** 2 + (c - pc) ** 2 < 36 for pr, pc in pad_cells):
            continue
        kind[r - 1:r + 3, c - 1:c + 3] = MD_RAMP
        kind[r:r + 2, c:c + 2] = MD_PAD
        feature |= dilate8(block)
        pad_cells.append((r, c))

    # subtle floor variation: raised 109 pads and 67 lows on plain floor
    interior = walk.copy()
    for _ in range(3):                    # keep 3 tiles from any wall/feature
        interior &= ~dilate8(~walk)
    interior &= ~feature
    cand = np.argwhere(interior)
    rng.shuffle(cand)
    placed = 0
    for r, c in cand:
        if placed >= 25:
            break
        rad = rng.randint(2, 3)
        blob = ((cc - c) ** 2 + (rr - r) ** 2) <= rad * rad
        if np.all(kind[blob] == FLOOR) and not np.any(feature[blob]):
            kind[blob] = PAD109 if placed % 2 == 0 else LOW67
            feature |= dilate8(blob)
            placed += 1

    # Raised pads must not sit against lake/lava ramps — that cracks the
    # walkable heightmap (109 plateau vs soft 67–85 skirts).
    near_bowl = dilate8(np.isin(kind, list(BOWL_KINDS)))
    kind[(kind == PAD109) & near_bowl] = FLOOR

    # biome index for every cell (nearest basin, normalized)
    dist = np.stack([
        ((cc - cx) / rx) ** 2 + ((rr - cz) / rz) ** 2
        for _, (cx, cz), (rx, rz) in BIOMES])
    biome_idx = np.argmin(dist, axis=0).astype(np.uint8)

    return kind, walk, conn_masks, lake_cells, lava_cells, pad_cells, biome_idx, ember_platforms


# =========================================================================
# Heightmap tile selection
# =========================================================================
def corner_targets(kind):
    lo = np.zeros((D, W)); hi = np.zeros((D, W))
    for k, (l, h) in KIND_RANGE.items():
        lo[kind == k] = l
        hi[kind == k] = h
    lop = np.zeros((D + 2, W + 2)); lop[1:-1, 1:-1] = lo
    raw = np.maximum.reduce([lop[:-1, :-1], lop[:-1, 1:], lop[1:, :-1], lop[1:, 1:]])
    return raw, lo, hi          # raw is (D+1, W+1)


def pack_hm(tid, fx, fy):
    return tid | (fx << 15) | (fy << 14)


def stamp_cliff_kit(kind, walk):
    """Stamp original Level1 cliff grammar: orth 2-tile stacks + diagonal stamps.

    Facing = direction toward walkable (low). Lower band uses 167/235 with
    path 43; upper uses matching-flip 166/217 with path 9. Corners use
    182/181/180 with matched flips. Diagonals are stamped before orth stacks
    so 167 never sits adjacent to 235 (matches original Level1).
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
    TIP_FLOOR = {
        "SW": ("N", "E"), "SE": ("N", "W"), "NW": ("S", "E"), "NE": ("S", "W"),
    }

    def inb(r, c):
        return 0 <= r < D and 0 <= c < W

    def lows_at(r, c):
        out = []
        for name, dr, dc in CARD:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and walk[nr, nc]:
                out.append(name)
        return out

    # Classify every lower-band cell before writing any tiles.
    classes = {}  # (r,c) -> ("diag", tip) | ("orth", facing)
    for r, c in zip(*np.nonzero(kind == CRAMP_L)):
        lows = lows_at(r, c)
        key = frozenset(lows)
        if len(lows) == 2 and key in LOW_TO_TIP:
            classes[(r, c)] = ("diag", LOW_TO_TIP[key])
        elif len(lows) >= 1:
            # Prefer the low side whose opposite neighbor is the upper/cliff band
            facing = lows[0]
            for f in lows:
                odr, odc = STEP[OPP[f]]
                nr, nc = r + odr, c + odc
                if inb(nr, nc) and kind[nr, nc] in (CRAMP_U, CLIFF):
                    facing = f
                    break
            classes[(r, c)] = ("orth", facing)
        else:
            # No cardinal floor (outer tip / degenerate): face nearest walkable.
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

    # Also promote orth cells that sit at a concave floor corner via 8-neighbors:
    # if two adjacent orth facings would meet as 167|235, upgrade to diagonal.
    for (r, c), (typ, facing) in list(classes.items()):
        if typ != "orth":
            continue
        lows = set(lows_at(r, c))
        # Re-check with diagonal walkable corners: floor SE of cell etc.
        for tip, (fx, fy) in CLIFF_DIAG.items():
            floor_dirs = TIP_FLOOR[tip]
            if all(
                any(
                    inb(r + STEP[d][0] * k, c + STEP[d][1] * k)
                    and walk[r + STEP[d][0] * k, c + STEP[d][1] * k]
                    for k in (1,)
                )
                for d in floor_dirs
            ) and not any(
                inb(r + STEP[d][0], c + STEP[d][1]) and walk[r + STEP[d][0], c + STEP[d][1]]
                for d in TIP_CLIFF[tip]
            ):
                # both floor-side cardinals are walkable and cliff-sides are not
                if set(floor_dirs).issubset(lows) or len(lows) == 0:
                    classes[(r, c)] = ("diag", tip)
                    break

    # Pass 1: diagonal stamps (181 / 180 / 182)
    for (r, c), (typ, tip) in classes.items():
        if typ != "diag":
            continue
        fx, fy = CLIFF_DIAG[tip]
        hm[r, c] = pack_hm(181, fx, fy)
        assigned[r, c] = True
        # Outer diagonal (both cliff dirs) is where original places 180.
        cliff_dirs = TIP_CLIFF[tip]
        dr = STEP[cliff_dirs[0]][0] + STEP[cliff_dirs[1]][0]
        dc = STEP[cliff_dirs[0]][1] + STEP[cliff_dirs[1]][1]
        ur, uc = r + dr, c + dc
        if inb(ur, uc) and kind[ur, uc] in (CRAMP_U, CLIFF) and not assigned[ur, uc]:
            hm[ur, uc] = pack_hm(180, fx, fy)
            assigned[ur, uc] = True
        for name in cliff_dirs:
            adr, adc = STEP[name]
            for dist in (1, 2):
                ur, uc = r + adr * dist, c + adc * dist
                if not inb(ur, uc) or assigned[ur, uc]:
                    continue
                if kind[ur, uc] in (CRAMP_U, CLIFF):
                    hm[ur, uc] = pack_hm(180, fx, fy)
                    assigned[ur, uc] = True
                    break
        for name in TIP_FLOOR[tip]:
            adr, adc = STEP[name]
            fr, fc = r + adr, c + adc
            if not inb(fr, fc) or assigned[fr, fc]:
                continue
            if kind[fr, fc] in (CRAMP_L, FLOOR, LOW67, PAD109):
                hm[fr, fc] = pack_hm(182, fx, fy)
                assigned[fr, fc] = True

    # Pass 2: orth stacks on remaining lower-band cells
    for (r, c), (typ, facing) in classes.items():
        if assigned[r, c] or typ != "orth":
            continue
        if facing not in CLIFF_ORTH:
            facing = "S"
        lo_tid, up_tid, fx, fy = CLIFF_ORTH[facing]
        hm[r, c] = pack_hm(lo_tid, fx, fy)
        assigned[r, c] = True
        odr, odc = STEP[OPP[facing]]
        ur, uc = r + odr, c + odc
        if inb(ur, uc) and kind[ur, uc] in (CRAMP_U, CLIFF) and not assigned[ur, uc]:
            hm[ur, uc] = pack_hm(up_tid, fx, fy)
            assigned[ur, uc] = True

    # Pass 3: any leftover CRAMP_L (should be rare)
    for r, c in zip(*np.nonzero((kind == CRAMP_L) & ~assigned)):
        lows = lows_at(r, c)
        facing = lows[0] if lows else "S"
        lo_tid, up_tid, fx, fy = CLIFF_ORTH.get(facing, CLIFF_ORTH["S"])
        hm[r, c] = pack_hm(lo_tid, fx, fy)
        assigned[r, c] = True
        odr, odc = STEP[OPP[facing]]
        ur, uc = r + odr, c + odc
        if inb(ur, uc) and kind[ur, uc] in (CRAMP_U, CLIFF) and not assigned[ur, uc]:
            hm[ur, uc] = pack_hm(up_tid, fx, fy)
            assigned[ur, uc] = True

    # Pass 4: remaining CRAMP_U — upper tile facing walkable / lower band
    for r, c in zip(*np.nonzero((kind == CRAMP_U) & ~assigned)):
        facing = None
        for name, dr, dc in CARD:
            nr, nc = r + dr, c + dc
            if inb(nr, nc) and (walk[nr, nc] or kind[nr, nc] == CRAMP_L):
                facing = name
                break
        if facing is None:
            facing = "S"
        _, up_tid, fx, fy = CLIFF_ORTH[facing]
        hm[r, c] = pack_hm(up_tid, fx, fy)
        assigned[r, c] = True

    hm[kind == CLIFF] = 0
    assigned[kind == CLIFF] = True

    # Repair lower orth tiles whose low edge does not face walkable floor.
    for r, c in zip(*np.nonzero(kind == CRAMP_L)):
        tid = int(hm[r, c]) & 0x0FFF
        if tid not in (167, 235):
            continue
        lows = lows_at(r, c)
        if not lows:
            continue
        # Prefer unique floor side; else the side opposite upper/cliff band.
        if len(lows) == 1:
            facing = lows[0]
        else:
            facing = lows[0]
            for f in lows:
                odr, odc = STEP[OPP[f]]
                nr, nc = r + odr, c + odc
                if inb(nr, nc) and kind[nr, nc] in (CRAMP_U, CLIFF):
                    facing = f
                    break
                # Also accept an already-stamped upper kit tile as the cliff side.
                if inb(nr, nc) and (int(hm[nr, nc]) & 0x0FFF) in (166, 217, 180, 0):
                    facing = f
                    break
        lo_tid, up_tid, fx, fy = CLIFF_ORTH[facing]
        hm[r, c] = pack_hm(lo_tid, fx, fy)
        odr, odc = STEP[OPP[facing]]
        ur, uc = r + odr, c + odc
        if inb(ur, uc) and kind[ur, uc] in (CRAMP_U, CLIFF):
            ut = int(hm[ur, uc]) & 0x0FFF
            if ut in (166, 217, 0, 180):
                hm[ur, uc] = pack_hm(up_tid, fx, fy)

    # Collapse illegal double-thick lower orth stacks into lower+upper pairs.
    ids = hm & 0x0FFF
    for r in range(D - 1):
        for c in range(W):
            a, b = int(ids[r, c]), int(ids[r + 1, c])
            if a != b or a not in (167, 235):
                continue
            if kind[r, c] != CRAMP_L or kind[r + 1, c] != CRAMP_L:
                continue
            # Northern cell becomes the matching upper tile.
            fx = (int(hm[r + 1, c]) >> 15) & 1
            fy = (int(hm[r + 1, c]) >> 14) & 1
            up = 217 if a == 235 else 166
            hm[r, c] = pack_hm(up, fx, fy)
    for r in range(D):
        for c in range(W - 1):
            a, b = int(ids[r, c]), int(ids[r, c + 1])
            if a != b or a not in (167, 235):
                continue
            if kind[r, c] != CRAMP_L or kind[r, c + 1] != CRAMP_L:
                continue
            fx = (int(hm[r, c]) >> 15) & 1
            fy = (int(hm[r, c]) >> 14) & 1
            # If both face E (fx=0), the western cell is toward cliff → make it upper.
            # If both face W (fx=1), the eastern cell is toward cliff → make it upper.
            up = 166
            if fx == 0:
                hm[r, c] = pack_hm(up, fx, fy)
            else:
                hm[r, c + 1] = pack_hm(up, fx, fy)
            ids = hm & 0x0FFF

    # Final cleanup: any remaining 167|235 adjacency → convert to 181.
    for _ in range(2):
        ids = hm & 0x0FFF
        for r in range(D):
            for c in range(W - 1):
                a, b = int(ids[r, c]), int(ids[r, c + 1])
                if {a, b} != {167, 235}:
                    continue
                for cc in (c, c + 1):
                    if kind[r, cc] != CRAMP_L:
                        continue
                    lows = frozenset(lows_at(r, cc))
                    tip = LOW_TO_TIP.get(lows, "SW")
                    fx, fy = CLIFF_DIAG[tip]
                    hm[r, cc] = pack_hm(181, fx, fy)
                    break
        for r in range(D - 1):
            for c in range(W):
                a, b = int(ids[r, c]), int(ids[r + 1, c])
                if {a, b} != {167, 235}:
                    continue
                for rr in (r, r + 1):
                    if kind[rr, c] != CRAMP_L:
                        continue
                    lows = frozenset(lows_at(rr, c))
                    tip = LOW_TO_TIP.get(lows, "SW")
                    fx, fy = CLIFF_DIAG[tip]
                    hm[rr, c] = pack_hm(181, fx, fy)
                    break

    return hm, assigned


def select_hm(kind, walk, stats, rng):
    hm_grid, cliff_assigned = stamp_cliff_kit(kind, walk)
    raw, lo, hi = corner_targets(kind)
    cache = {}
    flip_combos = ((0, 0), (1, 0), (0, 1), (1, 1))
    for r in range(D):
        for c in range(W):
            if cliff_assigned[r, c]:
                continue
            k = kind[r, c]
            if k in FLAT_TILE:
                hm_grid[r, c] = FLAT_TILE[k]
                continue
            if k not in RAMP_CANDIDATES:
                # Should not happen for cliff bands (already stamped); leave 0.
                continue
            l, h = KIND_RANGE[k]
            tgt = tuple(int(min(max(raw[rr, cx], l), h))
                        for rr, cx in ((r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)))
            key = (k, tgt)
            got = cache.get(key)
            if got is None:
                te = (0.5 * (tgt[0] + tgt[1]), 0.5 * (tgt[2] + tgt[3]),
                      0.5 * (tgt[0] + tgt[2]), 0.5 * (tgt[1] + tgt[3]))
                best, best_err = None, 1e18
                for tid in RAMP_CANDIDATES[k]:
                    for fx, fy in flip_combos:
                        cor, edg, _ = stats[(tid, fx, fy)]
                        err = sum((a - b) ** 2 for a, b in zip(cor, tgt))
                        err += 0.6 * sum((a - b) ** 2 for a, b in zip(edg, te))
                        if err < best_err:
                            best_err, best = err, (tid, fx, fy)
                got = pack_hm(best[0], best[1], best[2])
                cache[key] = got
            hm_grid[r, c] = got
    # Cliff stamping can mark floor-adjacent cells; never leave kit tiles on flats.
    for r in range(D):
        for c in range(W):
            k = kind[r, c]
            if k in FLAT_TILE:
                hm_grid[r, c] = FLAT_TILE[k]
    repair_walkable_seams(hm_grid, kind, cliff_assigned, stats)
    return hm_grid


def repair_walkable_seams(hm_grid, kind, cliff_assigned, stats):
    """Re-pick bowl ramps that crack against neighboring walkable flats."""
    flip_combos = ((0, 0), (1, 0), (0, 1), (1, 1))

    def edge_means(v):
        tid, fx, fy = v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1
        return stats[(tid, fx, fy)][1]  # N,S,W,E

    walkish = np.isin(kind, list(WALKABLE_KINDS | BOWL_KINDS | {MD_RAMP, MD_PAD}))
    for _ in range(3):
        fixed = 0
        pairs = []
        for r in range(D):
            for c in range(W - 1):
                if walkish[r, c] and walkish[r, c + 1] \
                        and not cliff_assigned[r, c] and not cliff_assigned[r, c + 1]:
                    e0 = edge_means(int(hm_grid[r, c]))[3]
                    e1 = edge_means(int(hm_grid[r, c + 1]))[2]
                    if abs(e0 - e1) >= 30:
                        pairs.append(("ew", r, c, e0, e1))
        for r in range(D - 1):
            for c in range(W):
                if walkish[r, c] and walkish[r + 1, c] \
                        and not cliff_assigned[r, c] and not cliff_assigned[r + 1, c]:
                    e0 = edge_means(int(hm_grid[r, c]))[1]
                    e1 = edge_means(int(hm_grid[r + 1, c]))[0]
                    if abs(e0 - e1) >= 30:
                        pairs.append(("ns", r, c, e0, e1))

        for orient, r, c, e0, e1 in pairs:
            if orient == "ew":
                cells = ((r, c, 3, e1), (r, c + 1, 2, e0))
            else:
                cells = ((r, c, 1, e1), (r + 1, c, 0, e0))
            for rr, cc, want_edge, neigh_h in cells:
                k = kind[rr, cc]
                if k not in RAMP_CANDIDATES:
                    continue
                best, best_err = int(hm_grid[rr, cc]), 1e18
                for tid in RAMP_CANDIDATES[k]:
                    for fx, fy in flip_combos:
                        edg = stats[(tid, fx, fy)][1]
                        err = (edg[want_edge] - neigh_h) ** 2
                        if err < best_err:
                            best_err = err
                            best = pack_hm(tid, fx, fy)
                # If no ramp matches the flat neighbor, collapse soft to shelf flat.
                if best_err >= 30 * 30 and k in (LK_SOFT, LV_SOFT):
                    best = FLAT_TILE[LK_SHELF]
                    best_err = 0
                    kind[rr, cc] = LK_SHELF if k == LK_SOFT else LV_BED
                if best != hm_grid[rr, cc]:
                    hm_grid[rr, cc] = best
                    fixed += 1
                break
        if fixed == 0:
            break


# =========================================================================
# Texture layer
# =========================================================================
def paint_textures(kind, hm_grid, biome_idx, hm2tex, rng):
    tex = np.zeros((D, W), np.uint16)
    near_lake = dilate8(dilate8(dilate8(np.isin(kind, list((LK_SOFT, LK_SHELF, LK_DEEP, LK_BED))))))

    def weave(options):
        return options[rng.randrange(len(options))]

    for r in range(D):
        for c in range(W):
            k = kind[r, c]
            b = biome_idx[r, c]
            v = int(hm_grid[r, c])
            key = (v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1)
            # Original cliff-kit textures always win when a kit hm tile is present
            if key in CLIFF_TEX:
                tex[r, c] = CLIFF_TEX[key]
                continue
            if key[0] in (166, 167, 180, 181, 182, 217, 235):
                tex[r, c] = hm2tex.get(key, weave(WALL_FALLBACK))
                continue
            if k in (FLOOR, LOW67, PAD109):
                if b == 0:
                    tex[r, c] = weave(SOUTH_GRASS)
                elif b == 1:
                    tex[r, c] = weave(JUNGLE_MIX) if rng.random() < 0.25 else weave(SOUTH_GRASS)
                elif b == 2:
                    tex[r, c] = weave(NORTH_GRASS_FLIP)
                elif b == 3:
                    tex[r, c] = weave(CRYSTAL_MIX) if rng.random() < 0.4 else weave(SOUTH_GRASS)
                else:
                    tex[r, c] = weave(NEST_WATERSIDE) if near_lake[r, c] else weave(SOUTH_GRASS)
            elif k == CLIFF:
                tex[r, c] = weave(EMBER_MESA) if b == 2 else weave(SOUTH_GRASS)
            elif k in (MD_RAMP, MD_PAD):
                tex[r, c] = hm2tex.get(key, weave(WALL_FALLBACK))
            elif k == LK_BED:
                tex[r, c] = weave(NEST_WATERSIDE if b == 4 else WATER_BED)
            elif k in (LK_DEEP, LK_SHELF):
                tex[r, c] = weave(SHORE)
            elif k == LK_SOFT:
                tex[r, c] = weave(SHORE) if rng.random() < 0.5 else weave(SOUTH_GRASS)
            elif k == LV_BED:
                tex[r, c] = weave(LAVA_BED)
            elif k == LV_SOFT:
                tex[r, c] = weave(LAVA_FRINGE)
    return tex


# =========================================================================
# Items
# =========================================================================
def px(col, row):
    return col * TILE_PX + 16, row * TILE_PX + 16


class Placer:
    def __init__(self, rng, bfs_set, kind, biome_idx):
        self.rng = rng
        self.bfs = bfs_set
        self.kind = kind
        self.biome_idx = biome_idx
        self.items = []                     # (x, z, type, (p0,p1,p2,p3), pro_only)
        self.occupied = set()
        self.protected = []                 # (col,row) with 3-tile clearance
        self.by_cat = {"enemy": [], "powerup": [], "tree": [], "scenery": []}

    def cell_free(self, col, row, cat, nn):
        if (col, row) in self.occupied:
            return False
        for pc, pr in self.protected:
            if (col - pc) ** 2 + (row - pr) ** 2 < 3 ** 2 and cat != "special":
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

    def scatter(self, n, cells, itype, parm_fn, cat, nn, pro_only=False,
                walk_only=True):
        placed = 0
        attempts = 0
        cells = list(cells)
        while placed < n and attempts < n * 600 and cells:
            attempts += 1
            row, col = cells[self.rng.randrange(len(cells))]
            if walk_only and (row, col) not in self.bfs:
                continue
            if self.kind[row, col] not in WALKABLE_KINDS:
                continue
            if not self.cell_free(col, row, cat, nn):
                continue
            self.add(col, row, itype, parm_fn(placed), cat, pro_only=pro_only)
            placed += 1
        return placed


def nearest_ok(placer, col, row, radius=12):
    """Nearest BFS-walkable plain-floor cell to (col,row)."""
    best, bd = None, 1e18
    for r in range(max(0, row - radius), min(D, row + radius + 1)):
        for c in range(max(0, col - radius), min(W, col + radius + 1)):
            if (r, c) in placer.bfs and placer.kind[r, c] in WALKABLE_KINDS \
                    and placer.cell_free(c, r, "special", 0):
                d = (c - col) ** 2 + (r - row) ** 2
                if d < bd:
                    bd, best = d, (c, r)
    if best is None:
        raise SystemExit(f"no walkable cell near ({col},{row})")
    return best


def build_items(rng, kind, hm_grid, mean_map, biome_idx, conn_masks,
                lake_cells, lava_cells, pad_cells, stats, ember_platforms):
    # BFS over walkable-ish cells (cliffs excluded). Lava is traversable but damaging.
    path9 = np.isin(kind, (CLIFF, CRAMP_L, CRAMP_U))
    open_walk = (~path9) & (mean_map <= 120.0)

    # Start on an Ember jump platform (first reserved island).
    if ember_platforms:
        sr, sc = ember_platforms[0]
    else:
        bx, bz = BIOMES[EMBER_BIOME][1]
        sc, sr = bx, bz
    seed = None
    for rad in range(0, 25):
        for r in range(sr - rad, sr + rad + 1):
            for c in range(sc - rad, sc + rad + 1):
                if 0 <= r < D and 0 <= c < W and open_walk[r, c] \
                        and kind[r, c] in WALKABLE_KINDS:
                    seed = (r, c)
                    break
            if seed:
                break
        if seed:
            break
    if seed is None:
        raise SystemExit("no walkable start cell")
    bfs = set()
    dq = deque([seed])
    bfs.add(seed)
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < D and 0 <= nc < W and (nr, nc) not in bfs and open_walk[nr, nc]:
                bfs.add((nr, nc))
                dq.append((nr, nc))

    P = Placer(rng, bfs, kind, biome_idx)

    biome_cells = [[] for _ in BIOMES]
    for r, c in bfs:
        if kind[r, c] in WALKABLE_KINDS:
            biome_cells[biome_idx[r, c]].append((r, c))
    conn_cells = []
    for m in conn_masks:
        cells = [(r, c) for r, c in zip(*np.nonzero(m))
                 if (r, c) in bfs and kind[r, c] in WALKABLE_KINDS]
        conn_cells.append(cells)

    # ---------------- start / eggs / portals -----------------------------
    scol, srow = seed[1], seed[0]
    egg_anchor = {}
    egg_spots = {}
    EGG_MIN_DIST = 14          # tiles — eggs within a biome stay fairly far apart

    def place_egg_cluster(species, acol, arow, n_floor=5):
        acol, arow = nearest_ok(P, acol, arow)
        egg_anchor[species] = (acol, arow)
        spots = []
        tries = 0
        while len(spots) < n_floor and tries < 8000:
            tries += 1
            ang = rng.random() * math.tau
            rad = rng.uniform(EGG_MIN_DIST, EGG_MIN_DIST + 8)
            c = int(acol + math.cos(ang) * rad)
            r = int(arow + math.sin(ang) * rad)
            if not (0 <= r < D and 0 <= c < W):
                continue
            if (r, c) not in P.bfs or kind[r, c] not in WALKABLE_KINDS:
                continue
            if not P.cell_free(c, r, "special", 0):
                continue
            if any((c - oc) ** 2 + (r - orow) ** 2 < EGG_MIN_DIST ** 2
                   for oc, orow in spots):
                continue
            spots.append((c, r))
        if len(spots) < n_floor:
            # fallback: looser radius but keep min spacing
            tries = 0
            while len(spots) < n_floor and tries < 8000:
                tries += 1
                if not biome_cells[species if species < 5 else 4]:
                    break
                r, c = biome_cells[species if species < 4 else 4][
                    rng.randrange(len(biome_cells[species if species < 4 else 4]))]
                c, r = int(c), int(r)
                if (r, c) not in P.bfs or kind[r, c] not in WALKABLE_KINDS:
                    continue
                if not P.cell_free(c, r, "special", 0):
                    continue
                if any((c - oc) ** 2 + (r - orow) ** 2 < EGG_MIN_DIST ** 2
                       for oc, orow in spots + [(acol, arow)]):
                    continue
                spots.append((c, r))
        if len(spots) < n_floor:
            raise SystemExit(f"could not cluster eggs for species {species} "
                             f"(got {len(spots)})")
        for c, r in spots:
            P.add(c, r, IT_EGG, (species, 0, 0, 1), "special", protect=True)
        egg_spots[species] = spots

    # species 0..3 near each biome center (auto-snapped)
    for sp, (name, (cx, cz), _) in enumerate(BIOMES[:4]):
        place_egg_cluster(sp, cx, cz)
    # species 4: 2 on nest pads, 3 on floor
    pads_for_eggs = pad_cells[:2]
    for pr, pc in pads_for_eggs:
        P.add(pc, pr, IT_EGG, (4, 0, 0, 1), "special", protect=True)
    nx, nz = BIOMES[4][1]
    place_egg_cluster(4, nx, nz, n_floor=3)

    # start: aim toward Ember's egg cluster (species 2)
    dcol = egg_anchor[2][0] - scol
    drow = egg_anchor[2][1] - srow
    aim = round(math.atan2(-dcol, -drow) / (math.tau / 8)) % 8
    P.add(scol, srow, IT_START, (aim, 0, 0, 0), "special", protect=True)

    # portals near egg clusters (auto-snap)
    portal_biomes = [0, 1, 2, 4]
    for pn, bi in enumerate(portal_biomes):
        ac, ar = egg_anchor[bi]
        c, r = nearest_ok(P, ac + 6, ar + 4)
        P.add(c, r, IT_PORTAL, (pn, 0, 0, 0), "special", protect=True)

    # ---------------- water / lava / step stones -------------------------
    # Water only in the deep basin — never on soft fringe or flat floor.
    DEEP_BASIN = {LK_BED, LK_DEEP}

    def water_item_ok(r, c):
        """In-game water patch is 8x8 tiles; must sit entirely in bed|deep."""
        r0, r1 = r - 3, r + 5
        c0, c1 = c - 3, c + 5
        if r0 < 0 or c0 < 0 or r1 > D or c1 > W:
            return False
        patch = kind[r0:r1, c0:c1]
        return bool(np.all(np.isin(patch, list(DEEP_BASIN))))

    def patch_points(bed):
        cells = np.argwhere(bed)
        if len(cells) == 0:
            return []
        r0, c0 = cells.min(axis=0)
        r1, c1 = cells.max(axis=0)
        bedset = {(r, c) for r, c in cells}
        pts = set()
        for gr in range(r0 + 3, r1 + 1, 8):
            for gc in range(c0 + 3, c1 + 1, 8):
                best = min(bedset, key=lambda rc: (rc[0] - gr) ** 2 + (rc[1] - gc) ** 2)
                pts.add(best)
        pts.add(tuple(cells.mean(axis=0).astype(int)))
        return sorted(p for p in pts if p in bedset)

    for bi, bed in lake_cells:
        for r, c in patch_points(bed):
            if (c, r) not in P.occupied and water_item_ok(r, c):
                P.add(c, r, IT_WATER, (0, 0, 0, 0), "special")
            # if 8x8 won't fit, skip — never paint water onto flat surroundings

    for bed, fire in lava_cells:
        for r, c in patch_points(bed):
            if (c, r) not in P.occupied and kind[r, c] == LV_BED:
                P.add(c, r, IT_LAVA, (0, 0, 0, 2 if fire else 0), "special")

    # Step-stone items on Ember jump platforms (not packed on lava).
    for pr, pc in ember_platforms[1:]:  # skip start island
        if (pc, pr) not in P.occupied and kind[pr, pc] in WALKABLE_KINDS:
            P.add(pc, pr, IT_STEPSTONE, (0, 0, 0, 0), "special")

    # ---------------- powerups -------------------------------------------
    POW_PLAN = [
        (0, [POW_HEALTH] * 3 + [POW_LASER] * 3 + [POW_HEAT] * 2 + [POW_SHIELD] + [POW_TRI]),
        (1, [POW_LASER] * 2 + [POW_TRI] * 2 + [POW_SONIC] * 2 + [POW_HEALTH] + [POW_HEAT]),
        # Ember is mostly lava — only a few platform pickups fit.
        (2, [POW_HEALTH, POW_LASER, POW_SHIELD]),
        (3, [POW_LASER] * 2 + [POW_TRI] * 2 + [POW_SONIC] * 2 + [POW_HEAT] + [POW_NUKE]),
        (4, [POW_HEALTH] * 2 + [POW_LASER] * 2 + [POW_SONIC] * 2 + [POW_HEAT] + [POW_NUKE]),
    ]
    for bi, kinds_list in POW_PLAN:
        lst = list(kinds_list)
        rng.shuffle(lst)
        nn = 3.0 if bi == EMBER_BIOME else 6.0
        got = P.scatter(len(lst), biome_cells[bi], IT_POWERUP,
                        lambda i, l=lst: (l[i], 0, 0, 0), "powerup", nn)
        if got < len(lst):
            raise SystemExit(f"powerup placement failed in biome {bi}")
    conn_pows = [POW_HEALTH, POW_HEAT, POW_TRI, POW_SHIELD, POW_NUKE,
                 POW_HEALTH, POW_HEAT, POW_TRI, POW_SHIELD, POW_TRI]
    for i, pw in enumerate(conn_pows):
        cells = conn_cells[i % len(conn_cells)]
        P.scatter(1, cells, IT_POWERUP, lambda _n, p=pw: (p, 0, 0, 0), "powerup", 6.0)

    # ---------------- enemies --------------------------------------------
    ENEMY_PLAN = [
        (0, [(IT_STEGO, 5), (IT_REX, 4)]),
        (1, [(IT_SPITTER, 20), (IT_PTERA, 4), (IT_STEGO, 4)]),
        (2, [(IT_PTERA, 8), (IT_TRICER, 8), (IT_REX, 5)]),
        (3, [(IT_SPITTER, 26), (IT_REX, 10)]),
        (4, [(IT_PTERA, 12), (IT_REX, 10), (IT_SPITTER, 8), (IT_STEGO, 7), (IT_TRICER, 5)]),
    ]
    enemy_counts = {}
    for bi, plan in ENEMY_PLAN:
        for et, n in plan:
            got = P.scatter(n, biome_cells[bi], et, lambda _n: (0, 0, 0, 0), "enemy", 2.0)
            enemy_counts[(bi, et)] = got
            # pro extras: +30%
            P.scatter(max(1, round(n * 0.3)), biome_cells[bi], et,
                      lambda _n: (0, 0, 0, 0), "enemy", 2.0, pro_only=True)
    CONN_ENEMY = [IT_STEGO, IT_SPITTER, IT_PTERA, IT_REX, IT_REX]
    for i, cells in enumerate(conn_cells):
        P.scatter(rng.randint(2, 3), cells, CONN_ENEMY[i],
                  lambda _n: (0, 0, 0, 0), "enemy", 2.0)

    # pro extra powerups (+30% of 95 ~ 28)
    pro_pow = ([POW_LASER] * 6 + [POW_HEALTH] * 6 + [POW_HEAT] * 5 +
               [POW_TRI] * 4 + [POW_SONIC] * 4 + [POW_SHIELD] * 2 + [POW_NUKE])
    rng.shuffle(pro_pow)
    for i, pw in enumerate(pro_pow):
        P.scatter(1, biome_cells[i % 5], IT_POWERUP,
                  lambda _n, p=pw: (p, 0, 0, 0), "powerup", 6.0, pro_only=True)

    # ---------------- scenery --------------------------------------------
    def trees(bi, n, types, nn):
        P.scatter(n, biome_cells[bi], IT_TREE,
                  lambda _n: (rng.choice(types), 0, 0, 0), "tree", nn)

    trees(0, 33, (0, 1), 2.5)
    trees(1, 45, (0, 1, 4), 2.5)
    trees(2, 6, (1,), 3.5)
    trees(3, 24, (1, 4), 3.5)
    trees(4, 18, (0, 1, 4), 3.0)

    P.scatter(6, biome_cells[0], IT_MUSHROOM, lambda _n: (0, 0, 0, 0), "scenery", 2.5)
    P.scatter(8, biome_cells[0], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 2.5)
    P.scatter(10, biome_cells[1], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 2.5)
    P.scatter(4, biome_cells[2], IT_BUSH, lambda _n: (0, 0, 0, 1), "scenery", 3.0)
    P.scatter(12, biome_cells[3], IT_CRYSTAL,
              lambda i: (i % 3, 0, 0, 0), "scenery", 3.0)
    P.scatter(4, biome_cells[2], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.0)
    P.scatter(4, biome_cells[3], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.0)
    P.scatter(5, conn_cells[3], IT_ROLLBOULDER, lambda _n: (0, 0, 0, 0), "scenery", 2.0)

    # gas vent cluster in Nest Caldera
    nx, nz = BIOMES[4][1]
    gc, gr = nearest_ok(P, nx, nz)
    placed = 0
    for dc, dr in ((0, 0), (2, 1), (1, 3), (3, 3), (-1, 2)):
        c, r = gc + dc, gr + dr
        if 0 <= r < D and 0 <= c < W and (r, c) in P.bfs \
                and kind[r, c] in WALKABLE_KINDS and P.cell_free(c, r, "scenery", 0):
            P.add(c, r, IT_GASVENT, (0, 0, 0, 0), "scenery")
            placed += 1
        if placed >= 3:
            break

    return P, bfs, (scol, srow), enemy_counts, biome_cells


# =========================================================================
# Serialization
# =========================================================================
def pack_items(items):
    blob = struct.pack(">i", len(items))
    for x, z, t, p, _ in items:
        blob += struct.pack(">hhh4bHii", x, z, t, p[0], p[1], p[2], p[3], 0, 0, 0)
    return blob


def write_ter(path, tex, hm_full_orig, path_layer, items, orig):
    # remap heightmap tiles compactly (ascending original id keeps tile 0 -> 0)
    used = sorted({int(v) & 0x0FFF for v in np.unique(hm_full_orig)})
    assert len(used) <= 300, f"too many hm tiles: {len(used)}"
    remap = {o: n for n, o in enumerate(used)}
    lut = np.zeros(4096, np.uint16)
    for o, n in remap.items():
        lut[o] = n
    hm_out = (lut[hm_full_orig & 0x0FFF] | (hm_full_orig & 0xC000)).astype(">u2")
    hm_blob = b"".join(orig.hm_tiles[o].tobytes() for o in used)

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
    rgb = orig.tile_avg[np.asarray(tex) & 0x0FFF]           # (D,W,3)
    img = Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    idx = np.asarray(img, np.uint8)
    ncol = int(idx.max()) + 1
    pal = img.getpalette()[:ncol * 3]
    header = struct.pack("<BBBHHBHHHHBB",
                         0,          # id length
                         1,          # colormap type
                         9,          # RLE colormapped (same as shipped Map.tga)
                         0, ncol, 24,          # colormap: origin, length, 24-bit
                         0, D,       # x-origin, y-origin (matches original: 355)
                         W, D, 8,
                         0x20)       # descriptor: top-left origin
    palette = bytearray()
    for i in range(ncol):
        r, g, b = pal[i * 3:i * 3 + 3]
        palette += bytes((b, g, r))                          # TGA palettes are BGR
    body = bytearray()
    for row in idx:                                          # RLE per row
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


def write_preview(path, tex, kind, mean_map, items, orig):
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
        cx, cy = x // TILE_PX * S + S // 2, z // TILE_PX * S + S // 2
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
    # legend
    legend = Image.new("RGB", (img.width, 70), (16, 16, 16))
    ld = ImageDraw.Draw(legend)
    lx = 8
    ld.text((lx, 6), "S start   P portal   E egg (color=species)", fill="white")
    ld.text((lx, 22), "dots: red=rex brown=tricer purple=ptera olive=stego blue=spitter "
                      "yellow=powerup cyan=crystal", fill="white")
    ld.text((lx, 38), "squares: blue=water item orange=lava item", fill="white")
    out = Image.new("RGB", (img.width, img.height + 70))
    out.paste(img, (0, 0))
    out.paste(legend, (0, img.height))
    out.save(path)


# =========================================================================
# Validation
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
                   f"{w}x{d}, {n_hm} hm tiles, max tex id {(tex & 0xFFF).max()}, "
                   f"{num_items} items"))

    xs = [it[0] for it in items]
    sorted_ok = all(xs[i] <= xs[i + 1] for i in range(len(xs) - 1))
    starts = [it for it in items if it[2] == IT_START]
    eggs = [it for it in items if it[2] == IT_EGG]
    egg_by_sp = Counter(e[3] for e in eggs)
    eggs_ok = all(egg_by_sp.get(s, 0) == 5 for s in range(5)) and \
        all((e[6] & 1) == 1 for e in eggs)
    portals = sorted(it[3] for it in items if it[2] == IT_PORTAL)
    types_ok = all(0 <= it[2] <= 19 for it in items)
    border_ok = all(4 <= it[0] // 32 < W - 4 and 4 <= it[1] // 32 < D - 4 for it in items)
    checks.append(("2 item rules", sorted_ok and len(starts) == 1 and eggs_ok
                   and portals == [0, 1, 2, 3] and types_ok and border_ok,
                   f"sorted={sorted_ok} starts={len(starts)} eggs/sp={dict(egg_by_sp)} "
                   f"portals={portals}"))

    # per-cell stats LUTs
    uniq = np.unique(hm)
    mean_lut, eN, eS, eW, eE = {}, {}, {}, {}, {}
    for v in uniq:
        t = hm_tiles[v & 0x0FFF].astype(np.float64)
        if v & 0x8000:
            t = t[:, ::-1]
        if v & 0x4000:
            t = t[::-1, :]
        mean_lut[v] = t.mean()
        eN[v], eS[v], eW[v], eE[v] = t[0].mean(), t[-1].mean(), t[:, 0].mean(), t[:, -1].mean()

    def grid_of(dic):
        out = np.zeros((d, w))
        for v in uniq:
            out[hm == v] = dic[v]
        return out
    mean_g = grid_of(mean_lut)
    gN, gS, gW, gE = grid_of(eN), grid_of(eS), grid_of(eW), grid_of(eE)

    border = np.zeros((d, w), bool)
    border[:3, :] = border[-3:, :] = border[:, :3] = border[:, -3:] = True
    cont_ok = bool(np.all(pathl[border] == 9) and np.all(mean_g[border] == 255.0))
    walk_open = (pathl == 0) & (mean_g <= 120.0)
    # BFS from start
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
    checks.append(("3 containment", cont_ok and not edge_touch,
                   f"border path9+255={cont_ok}, floor touches edge={edge_touch}"))

    # 4 connectivity
    check_types = {IT_EGG, IT_PORTAL, IT_POWERUP, IT_CRYSTAL, IT_MUSHROOM} | set(ENEMY_TYPES)
    unreachable = []
    pad_egg_notes = []
    ramp_ids_224 = set()
    for v in uniq:
        t = hm_tiles[v & 0x0FFF]
        if t.max() >= 200 and t.min() <= 100:
            ramp_ids_224.add(int(v) & 0x0FFF)
    for it in items:
        if it[2] not in check_types:
            continue
        r, c = it[1] // 32, it[0] // 32
        if seen[r, c]:
            continue
        # nest-pad egg special case: flat ~224 pad with an adjacent mound ramp
        tile_mean = mean_g[r, c]
        if it[2] == IT_EGG and 200 <= tile_mean <= 240:
            has_ramp = False
            ramp_near_floor = False
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < d and 0 <= nc < w and (hm[nr, nc] & 0xFFF) in ramp_ids_224 \
                            and mean_g[nr, nc] < 200:
                        has_ramp = True
                        for dr2 in (-1, 0, 1):
                            for dc2 in (-1, 0, 1):
                                r2, c2 = nr + dr2, nc + dc2
                                if 0 <= r2 < d and 0 <= c2 < w and seen[r2, c2]:
                                    ramp_near_floor = True
            if has_ramp and ramp_near_floor:
                pad_egg_notes.append((c, r))
                continue
        unreachable.append((it[2], c, r))
    checks.append(("4 connectivity", not unreachable,
                   f"unreachable={unreachable[:6]} pad_eggs_ok={len(pad_egg_notes)}"))

    # 5 water/lava
    wl_ok = True
    detail5 = []
    for it in items:
        r, c = it[1] // 32, it[0] // 32
        tid = int(tex[r, c]) & 0x0FFF
        flags = attribs[tid][0] if tid < len(attribs) else 0
        if it[2] == IT_WATER:
            hmin = int(hm_tiles[hm[r, c] & 0xFFF].min()) * 4
            if hmin >= 210 or not (flags & TILE_ATTRIB_WATER):
                wl_ok = False
                detail5.append(("water", c, r, hmin, hex(flags)))
        elif it[2] == IT_LAVA:
            hmin = int(hm_tiles[hm[r, c] & 0xFFF].min()) * 4
            if hmin >= 305 or not (flags & TILE_ATTRIB_LAVA):
                wl_ok = False
                detail5.append(("lava", c, r, hmin, hex(flags)))
    checks.append(("5 water/lava", wl_ok, f"bad={detail5[:5]}"))

    # 6 spacing
    tiles_used = Counter((it[0] // 32, it[1] // 32) for it in items)
    dup = [k for k, v in tiles_used.items() if v > 1]
    en = [(it[0] / 32, it[1] / 32) for it in items if it[2] in ENEMY_TYPES]
    pw = [(it[0] / 32, it[1] / 32) for it in items if it[2] == IT_POWERUP]

    def min_nn(pts):
        if len(pts) < 2:
            return 1e9
        arr = np.array(pts)
        best = 1e9
        for i in range(len(arr)):
            dd = np.hypot(arr[:, 0] - arr[i, 0], arr[:, 1] - arr[i, 1])
            dd[i] = 1e9
            best = min(best, dd.min())
        return best
    nn_e, nn_p = min_nn(en), min_nn(pw)
    checks.append(("6 spacing", not dup and nn_e >= 1.0 and nn_p >= 5.0,
                   f"dup_tiles={len(dup)} enemyNN={nn_e:.2f} powerupNN={nn_p:.2f}"))

    # 7 heightmap continuity — only require seamless edges on walkable path-0
    # cells. Cliff/slope bands (path 9 / 43) use the original kit and may have
    # intentional grade changes against floor.
    walkable = (pathl == 0)
    bad_ew = (np.abs(gE[:, :-1] - gW[:, 1:]) >= 30) & walkable[:, :-1] & walkable[:, 1:]
    bad_ns = (np.abs(gS[:-1, :] - gN[1:, :]) >= 30) & walkable[:-1, :] & walkable[1:, :]
    n_bad = int(bad_ew.sum() + bad_ns.sum())
    checks.append(("7 continuity", n_bad == 0, f"cracked pairs={n_bad}"))

    floor_pct = 100.0 * np.count_nonzero(walk_open) / (w * d)
    cliff_pct = 100.0 * np.count_nonzero(mean_g == 255.0) / (w * d)
    stats = dict(floor_pct=floor_pct, cliff_pct=cliff_pct, items=num_items,
                 hm_tiles=n_hm, bfs_cells=int(seen.sum()))
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
    stats = tile_stats(orig)
    hm2tex = derive_hm_to_tex(orig)

    # sanity of the kit tiles we rely on
    for tid, want in ((0, 255), (6, 85), (7, 67), (8, 41), (5, 109), (31, 224)):
        got = orig.hm_tiles[tid].mean()
        assert abs(got - want) < 6, f"hm tile {tid}: mean {got} != {want}"

    print("building geometry ...")
    kind, walk, conn_masks, lake_cells, lava_cells, pad_cells, biome_idx, ember_platforms = \
        build_geometry(rng)

    print("selecting heightmap tiles ...")
    hm_grid = select_hm(kind, walk, stats, rng)
    mean_map = np.zeros((D, W))
    for v in np.unique(hm_grid):
        mean_map[hm_grid == v] = stats[(v & 0xFFF, (v >> 15) & 1, (v >> 14) & 1)][2]

    # Original cliff path grammar: lower band = 43 (slope), upper/cliff = 9.
    # Floor cells stamped with 182 keep path 0 so they stay walkable.
    hm_ids = hm_grid & 0x0FFF
    path_layer = np.zeros((D, W), np.uint16)
    path_layer[kind == CLIFF] = 9
    path_layer[kind == CRAMP_U] = 9
    path_layer[kind == CRAMP_L] = 43
    # Upper / corner kit tiles are solid even if they sit on the lower band mask.
    path_layer[np.isin(hm_ids, [166, 217, 180, 181])] = 9

    print("painting textures ...")
    tex = paint_textures(kind, hm_grid, biome_idx, hm2tex, rng)

    print("placing items ...")
    P, bfs, start, enemy_counts, biome_cells = build_items(
        rng, kind, hm_grid, mean_map, biome_idx, conn_masks,
        lake_cells, lava_cells, pad_cells, stats, ember_platforms)

    items_sorted = sorted(P.items, key=lambda it: (it[0], it[1]))
    normal_items = [it for it in items_sorted if not it[4]]
    pro_items = items_sorted

    OUT_TERRAIN.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    n_tiles = write_ter(OUT_TERRAIN / "Level1.ter", tex, hm_grid, path_layer,
                        normal_items, orig)
    write_ter(OUT_TERRAIN / "Level1Pro.ter", tex, hm_grid, path_layer,
              pro_items, orig)
    shutil.copyfile(ORIG_DIR / "Level1.trt", OUT_TERRAIN / "Level1.trt")
    write_map_tga(OUT_IMAGES / "Map.tga", tex, orig)
    write_preview(PREVIEW_PATH, tex, kind, mean_map, P.items, orig)

    # ------------------------------------------------------------ validate
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
    print(f"\n  [{'PASS' if ok8 else 'FAIL'}] 8 Map.tga format: {det8}")

    # per-biome enemy density report
    print("\n  biome enemy densities (normal, per 1000 walkable):")
    for bi, (name, _, _) in enumerate(BIOMES):
        walkable = len(biome_cells[bi])
        n_en = sum(1 for x, z, t, p, pro in P.items
                   if not pro and t in ENEMY_TYPES
                   and biome_idx[z // 32, x // 32] == bi)
        print(f"    {name:18s}: {n_en:4d} enemies / {walkable:5d} cells "
              f"= {1000 * n_en / max(1, walkable):.1f}")

    if not all_ok:
        print("\nVALIDATION FAILED")
        sys.exit(1)
    print("\nALL CHECKS PASSED")

    # ------------------------------------------------------ install bundle
    if BUNDLE.is_dir():
        (BUNDLE / "Terrain").mkdir(parents=True, exist_ok=True)
        (BUNDLE / "Images").mkdir(parents=True, exist_ok=True)
        for f in ("Level1.ter", "Level1Pro.ter", "Level1.trt"):
            shutil.copyfile(OUT_TERRAIN / f, BUNDLE / "Terrain" / f)
        shutil.copyfile(OUT_IMAGES / "Map.tga", BUNDLE / "Images" / "Map.tga")
        print(f"installed into {BUNDLE}")


if __name__ == "__main__":
    main()
