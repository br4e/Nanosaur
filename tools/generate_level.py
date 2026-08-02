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
    CLIFF: (255, 255), CRAMP_L: (85, 152), CRAMP_U: (152, 255),
    FLOOR: (85, 85), LOW67: (67, 67), PAD109: (109, 109),
    LK_SOFT: (67, 85), LK_SHELF: (67, 67), LK_DEEP: (41, 67), LK_BED: (41, 41),
    LV_SOFT: (67, 85), LV_BED: (67, 67),
    MD_RAMP: (85, 224), MD_PAD: (224, 224),
}
FLAT_TILE = {CLIFF: 0, FLOOR: 6, LOW67: 7, PAD109: 5,
             LK_SHELF: 7, LK_BED: 8, LV_BED: 7, MD_PAD: 31}
RAMP_CANDIDATES = {
    CRAMP_L: (235, 217, 167, 166, 182, 181, 180),
    CRAMP_U: (235, 217, 167, 166, 182, 181, 180),
    LK_SOFT: (22, 37, 48, 38, 23),
    LV_SOFT: (22, 37, 48, 38, 23),
    LK_DEEP: (24, 40, 39, 51, 35),
    MD_RAMP: (1, 32, 2, 17),
}
KIT_HM_IDS = sorted({t for v in RAMP_CANDIDATES.values() for t in v} | set(FLAT_TILE.values()))

WALKABLE_KINDS = {FLOOR, LOW67, PAD109}          # where regular items may go
BOWL_KINDS = {LK_SOFT, LK_SHELF, LK_DEEP, LK_BED, LV_SOFT, LV_BED}

# ------------------------------------------------------------ biome geometry
# (name, center(col,row), radii(rx,rz)) — compact arenas, ring pulled inward
BIOMES = [
    ("Emerald Shallows", (87, 231), (28, 28)),
    ("Fern Canyons",     (81, 140), (24, 34)),
    ("Ember Flats",      (135, 100), (38, 26)),
    ("Crystal Scar",     (188, 155), (26, 32)),
    ("Nest Caldera",     (180, 240), (30, 28)),
]
MESA = ((135, 180), (32, 38))
# short connector bridges between nearby basin edges
CONNECTORS = [
    ("Shallows->Fern",  [(85, 210), (82, 190), (80, 165)], (0.4, 0.7)),
    ("Fern->Ember",     [(95, 125), (110, 110), (120, 100)], (0.45,)),
    ("Ember->Crystal",  [(155, 95), (165, 115), (175, 135)], (0.35, 0.7)),
    ("Crystal->Nest",   [(190, 175), (188, 195), (182, 215)], (0.5,)),
    ("Nest->Shallows",  [(165, 245), (135, 245), (110, 235)], (0.35, 0.7)),
]
FERN_BLOBS = [(69, 122, 5), (91, 137, 5), (73, 157, 6), (89, 167, 4)]

# lakes: (biome_idx, col, row, rx, rz) — pulled toward biome centers
LAKES = [
    (0, 75, 221, 3, 2), (0, 97, 235, 3, 2), (0, 83, 245, 2, 2),
    (3, 180, 143, 2, 2), (3, 196, 165, 2, 2),
    (4, 176, 236, 4, 3),
]
# lava pools in Ember: (col,row,rx,rz, fireballs)
LAVA_POOLS = [
    (135, 93, 6, 4, True),      # the big field (gets step stones)
    (120, 91, 2, 2, True), (148, 91, 2, 2, True), (125, 108, 2, 2, True),
    (145, 111, 2, 2, True), (112, 101, 2, 2, False), (155, 98, 2, 2, True),
]
# nest pad complexes: top-left cell of the 2x2 pad (ring adds 1 cell around)
NEST_PADS = [(164, 248), (174, 256), (190, 250), (196, 236), (188, 226)]

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
        return bool(np.all(kind[mask] == FLOOR) and not np.any(feature[mask]))

    lake_cells = []
    for bi, cx, cz, rx, rz in LAKES:
        bed = ellipse(cx, cz, rx, rz)
        deep = dilate8(bed) & ~bed
        shelf = dilate8(dilate8(bed)) & ~bed & ~deep
        soft = dilate8(dilate8(dilate8(bed))) & ~bed & ~deep & ~shelf
        whole = bed | deep | shelf | soft
        if not ring_ok(whole):
            raise SystemExit(f"lake at ({cx},{cz}) does not fit on plain floor")
        kind[bed], kind[deep], kind[shelf], kind[soft] = LK_BED, LK_DEEP, LK_SHELF, LK_SOFT
        feature |= dilate8(dilate8(whole))
        lake_cells.append((bi, bed))

    lava_cells = []
    for cx, cz, rx, rz, fire in LAVA_POOLS:
        bed = ellipse(cx, cz, rx, rz)
        soft = dilate8(bed) & ~bed
        whole = bed | soft
        if not ring_ok(whole):
            raise SystemExit(f"lava at ({cx},{cz}) does not fit on plain floor")
        kind[bed], kind[soft] = LV_BED, LV_SOFT
        feature |= dilate8(dilate8(whole))
        lava_cells.append((bed, fire))

    pad_cells = []
    for c0, r0 in NEST_PADS:
        rows, cols = slice(r0 - 1, r0 + 3), slice(c0 - 1, c0 + 3)
        block = np.zeros((D, W), bool)
        block[rows, cols] = True
        if not ring_ok(block):
            raise SystemExit(f"nest pad at ({c0},{r0}) does not fit on plain floor")
        kind[rows, cols] = MD_RAMP
        kind[r0:r0 + 2, c0:c0 + 2] = MD_PAD
        feature |= dilate8(block)
        pad_cells.append((r0, c0))

    # subtle floor variation: raised 109 pads and 67 lows on plain floor
    interior = walk.copy()
    for _ in range(3):                    # keep 3 tiles from any wall/feature
        interior &= ~dilate8(~walk)
    interior &= ~feature
    cand = np.argwhere(interior)
    rng.shuffle(cand)
    placed = 0
    for r, c in cand:
        if placed >= 60:
            break
        rad = rng.randint(2, 3)
        blob = ((cc - c) ** 2 + (rr - r) ** 2) <= rad * rad
        if np.all(kind[blob] == FLOOR) and not np.any(feature[blob]):
            kind[blob] = PAD109 if placed % 2 == 0 else LOW67
            feature |= dilate8(blob)
            placed += 1

    # biome index for every cell (nearest basin, normalized)
    dist = np.stack([
        ((cc - cx) / rx) ** 2 + ((rr - cz) / rz) ** 2
        for _, (cx, cz), (rx, rz) in BIOMES])
    biome_idx = np.argmin(dist, axis=0).astype(np.uint8)

    return kind, walk, conn_masks, lake_cells, lava_cells, pad_cells, biome_idx


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


def select_hm(kind, stats, rng):
    raw, lo, hi = corner_targets(kind)
    hm_grid = np.zeros((D, W), np.uint16)   # full16 with ORIGINAL tile ids
    cache = {}
    flip_combos = ((0, 0), (1, 0), (0, 1), (1, 1))
    for r in range(D):
        for c in range(W):
            k = kind[r, c]
            if k in FLAT_TILE:
                hm_grid[r, c] = FLAT_TILE[k]
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
                got = best[0] | (best[1] << 15) | (best[2] << 14)
                cache[key] = got
            hm_grid[r, c] = got
    return hm_grid


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
            elif k in (CRAMP_L, CRAMP_U, MD_RAMP, MD_PAD):
                v = int(hm_grid[r, c])
                key = (v & 0x0FFF, (v >> 15) & 1, (v >> 14) & 1)
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
                lake_cells, lava_cells, pad_cells, stats):
    # BFS over walkable floor (path==0 is implied by kind; mean<=120)
    path9 = np.isin(kind, (CLIFF, CRAMP_L, CRAMP_U))
    open_walk = (~path9) & (mean_map <= 120.0)
    start_tile = nearest_start = None

    # start position: south part of Emerald Shallows
    sc, sr = 89, 241
    # BFS from a seed near start
    seed = None
    for rad in range(0, 20):
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

    def place_egg_cluster(species, acol, arow, n_floor=5):
        acol, arow = nearest_ok(P, acol, arow)
        egg_anchor[species] = (acol, arow)
        spots = []
        tries = 0
        while len(spots) < n_floor and tries < 4000:
            tries += 1
            ang = rng.random() * math.tau
            rad = rng.uniform(4, 11)
            c = int(acol + math.cos(ang) * rad)
            r = int(arow + math.sin(ang) * rad)
            if not (0 <= r < D and 0 <= c < W):
                continue
            if (r, c) not in P.bfs or kind[r, c] not in WALKABLE_KINDS:
                continue
            if not P.cell_free(c, r, "special", 0):
                continue
            if any((c - oc) ** 2 + (r - orow) ** 2 < 36 for oc, orow in spots):
                continue
            spots.append((c, r))
        if len(spots) < n_floor:
            raise SystemExit(f"could not cluster eggs for species {species}")
        for c, r in spots:
            P.add(c, r, IT_EGG, (species, 0, 0, 1), "special", protect=True)
        egg_spots[species] = spots

    # species 0..3 on floor in their biomes
    place_egg_cluster(0, 80, 223)
    place_egg_cluster(1, 77, 137)
    place_egg_cluster(2, 135, 103)
    place_egg_cluster(3, 188, 155)
    # species 4: 2 on nest pads, 3 on floor
    pads_for_eggs = pad_cells[:2]
    for pr, pc in pads_for_eggs:
        P.add(pc, pr, IT_EGG, (4, 0, 0, 1), "special", protect=True)
    place_egg_cluster(4, 176, 243, n_floor=3)
    egg_anchor[4] = (176, 243)

    # start (aim: forward = (-sin(aim*45), -cos(aim*45)) in (x,z))
    dcol = egg_anchor[0][0] - scol
    drow = egg_anchor[0][1] - srow
    aim = round(math.atan2(-dcol, -drow) / (math.tau / 8)) % 8
    P.add(scol, srow, IT_START, (aim, 0, 0, 0), "special", protect=True)

    # portals 0..3 in Shallows, Fern, Ember, Nest, near egg clusters
    portal_want = [(0, 97, 221), (1, 87, 152), (2, 150, 98), (4, 191, 243)]
    for pn, (bi, c, r) in enumerate(portal_want):
        c, r = nearest_ok(P, c, r)
        d = math.dist((c, r), egg_anchor[bi if bi != 4 else 4])
        P.add(c, r, IT_PORTAL, (pn, 0, 0, 0), "special", protect=True)

    # ---------------- water / lava / step stones -------------------------
    def patch_points(bed):
        """8-tile grid over the bed's bounding box, snapped to bed cells
        (an in-game water/lava patch is 8x8 tiles centered on the item)."""
        cells = np.argwhere(bed)
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
            if (c, r) not in P.occupied:
                P.add(c, r, IT_WATER, (0, 0, 0, 0), "special")

    big_bed = None
    for bed, fire in lava_cells:
        if big_bed is None or bed.sum() > big_bed.sum():
            big_bed = bed
        for r, c in patch_points(bed):
            if (c, r) not in P.occupied:
                P.add(c, r, IT_LAVA, (0, 0, 0, 2 if fire else 0), "special")

    # step stones across the big lava field (type 17)
    bcells = np.argwhere(big_bed)
    row_mid = int(bcells[:, 0].mean())
    cols = sorted({c for r, c in bcells if r == row_mid})
    stones = 0
    for c in cols[::2]:
        if stones >= 12:
            break
        if (c, row_mid) not in P.occupied:
            P.add(c, row_mid, IT_STEPSTONE, (0, 0, 0, 0), "special")
            stones += 1
    if stones < 6:     # widen if the center row was crowded
        for r, c in bcells[::3]:
            if stones >= 10:
                break
            if (c, r) not in P.occupied:
                P.add(c, r, IT_STEPSTONE, (0, 0, 0, 0), "special")
                stones += 1

    # ---------------- powerups -------------------------------------------
    POW_PLAN = [
        (0, [POW_HEALTH] * 4 + [POW_LASER] * 4 + [POW_HEAT] * 3 + [POW_SHIELD] + [POW_TRI]),
        (1, [POW_LASER] * 2 + [POW_TRI] * 2 + [POW_SONIC] * 2 + [POW_HEALTH] * 2 + [POW_HEAT]),
        (2, [POW_HEALTH] * 2 + [POW_LASER] * 2 + [POW_TRI] * 2 + [POW_SONIC] + [POW_NUKE, POW_SHIELD]),
        (3, [POW_LASER] * 3 + [POW_TRI] * 2 + [POW_SONIC] * 2 + [POW_HEAT] + [POW_NUKE, POW_SHIELD]),
        (4, [POW_HEALTH] * 3 + [POW_LASER] * 2 + [POW_SONIC] * 2 + [POW_HEAT] * 2 + [POW_NUKE]),
    ]
    for bi, kinds_list in POW_PLAN:
        lst = list(kinds_list)
        rng.shuffle(lst)
        got = P.scatter(len(lst), biome_cells[bi], IT_POWERUP,
                        lambda i, l=lst: (l[i], 0, 0, 0), "powerup", 6.0)
        if got < len(lst):
            raise SystemExit(f"powerup placement failed in biome {bi}")
    conn_pows = [POW_HEALTH, POW_HEAT, POW_TRI, POW_SHIELD, POW_NUKE,
                 POW_HEALTH, POW_HEAT, POW_TRI, POW_SHIELD, POW_TRI]
    for i, pw in enumerate(conn_pows):
        cells = conn_cells[i % len(conn_cells)]
        P.scatter(1, cells, IT_POWERUP, lambda _n, p=pw: (p, 0, 0, 0), "powerup", 6.0)

    # ---------------- enemies --------------------------------------------
    ENEMY_PLAN = [
        (0, [(IT_STEGO, 9), (IT_REX, 6)]),
        (1, [(IT_SPITTER, 33), (IT_PTERA, 7), (IT_STEGO, 6)]),
        (2, [(IT_PTERA, 13), (IT_TRICER, 13), (IT_REX, 9)]),
        (3, [(IT_SPITTER, 44), (IT_REX, 17)]),
        (4, [(IT_PTERA, 20), (IT_REX, 17), (IT_SPITTER, 13), (IT_STEGO, 11), (IT_TRICER, 9)]),
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

    trees(0, 55, (0, 1), 3.0)
    trees(1, 75, (0, 1, 4), 3.0)
    trees(2, 10, (1,), 4.0)
    trees(3, 40, (1, 4), 4.0)
    trees(4, 30, (0, 1, 4), 3.5)

    P.scatter(10, biome_cells[0], IT_MUSHROOM, lambda _n: (0, 0, 0, 0), "scenery", 3.0)
    P.scatter(12, biome_cells[0], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)
    P.scatter(14, biome_cells[1], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)
    P.scatter(6, biome_cells[2], IT_BUSH, lambda _n: (0, 0, 0, 1), "scenery", 4.0)
    P.scatter(16, biome_cells[3], IT_CRYSTAL,
              lambda i: (i % 3, 0, 0, 0), "scenery", 4.0)
    P.scatter(6, biome_cells[2], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 4.0)
    P.scatter(6, biome_cells[3], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 4.0)
    P.scatter(5, conn_cells[3], IT_ROLLBOULDER, lambda _n: (0, 0, 0, 0), "scenery", 2.0)

    # gas vent cluster in Nest Caldera
    gc, gr = nearest_ok(P, 184, 238)
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

    # 7 heightmap continuity
    p9 = (pathl == 9)
    bad_ew = (np.abs(gE[:, :-1] - gW[:, 1:]) >= 30) & ~p9[:, :-1] & ~p9[:, 1:]
    bad_ns = (np.abs(gS[:-1, :] - gN[1:, :]) >= 30) & ~p9[:-1, :] & ~p9[1:, :]
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
    kind, walk, conn_masks, lake_cells, lava_cells, pad_cells, biome_idx = \
        build_geometry(rng)

    print("selecting heightmap tiles ...")
    hm_grid = select_hm(kind, stats, rng)
    mean_map = np.zeros((D, W))
    for v in np.unique(hm_grid):
        mean_map[hm_grid == v] = stats[(v & 0xFFF, (v >> 15) & 1, (v >> 14) & 1)][2]

    path_layer = np.where(np.isin(kind, (CLIFF, CRAMP_L, CRAMP_U)), 9, 0).astype(np.uint16)

    print("painting textures ...")
    tex = paint_textures(kind, hm_grid, biome_idx, hm2tex, rng)

    print("placing items ...")
    P, bfs, start, enemy_counts, biome_cells = build_items(
        rng, kind, hm_grid, mean_map, biome_idx, conn_masks,
        lake_cells, lava_cells, pad_cells, stats)

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
