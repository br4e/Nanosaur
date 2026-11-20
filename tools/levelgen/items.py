"""Terrain item placement: portals, flat eggs, stepstones, hazards, enemies, and scenery."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .constants import (
    D,
    IT_BOULDER,
    IT_BUSH,
    IT_CRYSTAL,
    IT_EGG,
    IT_GASVENT,
    IT_LAVA,
    IT_MUSHROOM,
    IT_PORTAL,
    IT_POWERUP,
    IT_PTERA,
    IT_REX,
    IT_ROLLBOULDER,
    IT_SPITTER,
    IT_SPOREPOD,
    IT_START,
    IT_STEGO,
    IT_STEPSTONE,
    IT_TREE,
    IT_TRICER,
    IT_WATER,
    KIND_CLIFF,
    KIND_FLOOR,
    KIND_SPIRE_178,
    PATH_OPEN,
    POW_HEALTH,
    POW_HEAT,
    POW_LASER,
    POW_NUKE,
    POW_SHIELD,
    POW_SONIC,
    POW_TRI,
    TILE_PX,
    TREE_BAMBOO,
    TREE_CYPRESS,
    TREE_FERN,
    TREE_MAINPALM,
    TREE_PINEPALM,
    TREE_STICKPALM,
    W,
    ZONE_CANYON,
    ZONE_LAGOON,
    ZONE_LAVA,
    ZONE_MARSH,
)
from .geometry import dilate8, dilate_n, erode8


def px(col, row):
    """Return tile center in Oreo map pixel coordinates."""
    return col * TILE_PX + 16, row * TILE_PX + 16


class Placer:
    """Stateful spatial item placer with collision/clearance tracking."""

    def __init__(self, rng, bfs_set, world):
        self.rng = rng
        self.bfs = bfs_set
        self.world = world
        self.items = []  # (x, z, type, (p0,p1,p2,p3), pro_only)
        self.occupied = set()
        self.protected = []  # (col, row) with 3-tile clearance
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
        del nn  # spacing is enforced via cell_free before calling add
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
        del cat
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
    """Find the closest BFS-reachable free cell matching `allowed_mask` and `avoid` clearance."""
    del world
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
    """Cover 100% of `mask` tiles using 8x8 tile patches centered at (row, col)."""
    del min_gap
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    if not cells:
        return []
    uncovered = set(cells)
    chosen = []
    cell_set = set(cells)

    def covered_by_wide(cr, cc):
        return {
            (r, c)
            for r in range(cr - 3, cr + 5)
            for c in range(cc - 3, cc + 5)
            if (r, c) in uncovered
        }

    while uncovered:
        best_pt = None
        best_cov = set()
        sample = list(uncovered)
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
            cov = covered_by_wide(cr, cc)
            if len(cov) > len(best_cov):
                best_cov = cov
                best_pt = (cr, cc)
        if not best_pt or not best_cov:
            best_pt = next(iter(uncovered))
            best_cov = covered_by_wide(*best_pt) | {best_pt}
        chosen.append(best_pt)
        uncovered -= best_cov

    return chosen


def compute_flat_3x3_mask(world, path_layer):
    """Return mask of tiles whose 3x3 neighborhood is strictly flat, path==0, and non-hazard."""
    walk = world["walk"]
    tmin = world["tmin"]
    tmax = world["tmax"]
    is_water = world["is_water_bed"] | world["is_water_47"] | world["is_water_shore"] | world["is_ford_tile"]
    is_lava = world["is_lava_tile"] | world["is_lava_rim"]

    flat1 = (
        walk
        & (path_layer == PATH_OPEN)
        & (tmin == tmax)
        & ~is_water
        & ~is_lava
    )
    # Require all 9 tiles in 3x3 to be flat1 AND share the exact same elevation!
    flat3 = erode8(flat1)
    same_h = np.ones((D, W), dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.roll(np.roll(tmin, -dr, axis=0), -dc, axis=1)
            same_h &= (tmin == shifted)
    return flat3 & same_h


def build_items(rng, world, hm_grid, path_layer):
    """Populate all items across all 4 zones, enforcing strict flatness for portals and eggs."""
    del hm_grid
    walk = world["walk"]
    kind = world["kind"]
    zone = world["zone"]
    tmin = world["tmin"]
    tmax = world["tmax"]
    is_water_bed = world["is_water_bed"]
    is_water_47 = world["is_water_47"]
    is_water_shore = world["is_water_shore"]
    is_lava_tile = world["is_lava_tile"]
    is_lava_rim = world["is_lava_rim"]

    walk_open = (path_layer == PATH_OPEN) & (tmax <= 185)

    # Start in a safe grassy clearing in Zone 1 Emerald Marsh (NW, around col 88, row 135)
    start_c, start_r = 88, 134
    dry_marsh = (
        walk_open
        & (zone == ZONE_MARSH)
        & (kind == KIND_FLOOR)
        & ~dilate8(is_water_bed | is_water_47 | ~walk)
    )
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

    # Distance/clearance masks
    water_hazard = is_water_bed | is_water_47
    lava_hazard = is_lava_tile
    near_cliff_2 = dilate_n(~walk, 1, 8)  # cliff tiles are ~walk; dist_cliff >= 2 excludes the 1-tile ring around ~walk
    near_water_2 = dilate_n(water_hazard, 1, 8)
    near_lava_2 = dilate_n(lava_hazard, 1, 8)

    dry_mask = walk_open & ~water_hazard & ~lava_hazard
    safe_dry_mask = dry_mask & ~dilate8(~walk) & ~is_water_shore & ~is_lava_rim

    # Strict 3x3 flat mask for non-spire eggs (dist_cliff >= 2, dist_water >= 2, dist_lava >= 2)
    flat_3x3_mask = compute_flat_3x3_mask(world, path_layer)
    egg_flat_mask = flat_3x3_mask & ~near_cliff_2 & ~near_water_2 & ~near_lava_2
    # Strict flat mask for spire eggs (hm == 178, tmin == tmax == 178, path == 0)
    spire_flat_mask = (
        walk_open
        & (kind == KIND_SPIRE_178)
        & (tmin == 178)
        & (tmax == 178)
    )

    zone_dry = [[] for _ in range(4)]
    zone_flat = [[] for _ in range(4)]
    for r, c in bfs:
        if safe_dry_mask[r, c]:
            zone_dry[zone[r, c]].append((r, c))
            if kind[r, c] == KIND_FLOOR:
                zone_flat[zone[r, c]].append((r, c))

    # ---------------- Start (S) ----------------
    P.add(sc, sr, IT_START, (6, 0, 0, 0), "special", protect=True)

    # ---------------- Water & Lava Patches (100% coverage) ----------------
    water_need = is_water_bed | is_water_47
    water_centers = compute_patch_centers(water_need, min_gap=7)
    for r, c in water_centers:
        P.add(c, r, IT_WATER, (0, 0, 0, 0), "special", protect=False)

    # StepStone Bridges in Zone 2 Obsidian Lava Gorge
    bridge_points = []
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

    lava_centers = compute_patch_centers(is_lava_tile, min_gap=7)
    lava_cov = np.zeros((D, W), dtype=bool)
    for r, c in lava_centers:
        if (c, r) in P.occupied:
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

    # ---------------- Portals (P0..P3) on Strictly Flat Landmarks ----------------
    portal_coords = world["portal_coords"]
    p0_c, p0_r = nearest_dry_cell(
        P, world, *portal_coords[0],
        allowed_mask=flat_3x3_mask & world["marsh_p0_terrace"], radius=3,
    )
    P.add(p0_c, p0_r, IT_PORTAL, (0, 0, 0, 0), "special", protect=True)

    p1_c, p1_r = nearest_dry_cell(
        P, world, *portal_coords[1],
        allowed_mask=flat_3x3_mask & world["lava_portal_island"], radius=3,
    )
    P.add(p1_c, p1_r, IT_PORTAL, (1, 0, 0, 0), "special", protect=True)

    p2_c, p2_r = nearest_dry_cell(
        P, world, *portal_coords[2],
        allowed_mask=flat_3x3_mask & world["canyon_p2_terrace"], radius=3,
    )
    P.add(p2_c, p2_r, IT_PORTAL, (2, 0, 0, 0), "special", protect=True)

    p3_c, p3_r = nearest_dry_cell(
        P, world, *portal_coords[3],
        allowed_mask=flat_3x3_mask & world["lagoon_p3_terrace"], radius=3,
    )
    P.add(p3_c, p3_r, IT_PORTAL, (3, 0, 0, 0), "special", protect=True)

    # ---------------- Eggs (25 total: 5 per species 0..4, ALL strictly flat) ----------------
    egg_avoid = [(sc, sr), (p0_c, p0_r), (p1_c, p1_r), (p2_c, p2_r), (p3_c, p3_r)]

    def place_egg(species, col, row, mask=None, radius=16, min_sep=9):
        c, r = nearest_dry_cell(
            P, world, col, row,
            allowed_mask=(mask if mask is not None else egg_flat_mask),
            radius=radius,
            avoid=egg_avoid,
            clearance=min_sep,
        )
        P.add(c, r, IT_EGG, (species, 0, 0, 1), "special", protect=True)
        egg_avoid.append((c, r))
        return c, r

    # Species 0 (5 eggs in Emerald Marsh, strictly on flat_3x3_mask)
    place_egg(0, 105, 141, mask=egg_flat_mask & world["marsh_island"], radius=5, min_sep=4)
    place_egg(0, 88, 154, min_sep=8)
    place_egg(0, 100, 152, min_sep=8)
    place_egg(0, 117, 132, min_sep=8)
    place_egg(0, 108, 120, min_sep=8)

    # Species 1 (5 eggs in Obsidian Lava Gorge, strictly on flat_3x3_mask)
    place_egg(1, 155, 125, mask=egg_flat_mask & world["lava_north_island"], radius=5, min_sep=6)
    place_egg(1, 174, 141, mask=egg_flat_mask & world["lava_east_island"], radius=5, min_sep=6)
    place_egg(1, 143, 122, mask=egg_flat_mask & world["lava_overlook"], radius=5, min_sep=6)
    place_egg(1, 138, 144, min_sep=8)
    place_egg(1, 173, 154, min_sep=8)

    # Species 2 (5 eggs in Crystal Canyon, Oasis, and Secret Alcove, strictly on flat_3x3_mask)
    place_egg(2, 136, 178, min_sep=8)
    place_egg(2, 127, 189, min_sep=8)
    place_egg(2, 102, 192, min_sep=8)
    place_egg(2, 156, 190, mask=egg_flat_mask & world["secret_alcove"], radius=6, min_sep=6)
    place_egg(2, 148, 175, min_sep=8)

    # Species 3 (5 eggs in Ancient Nest Lagoon on nest terraces & strictly flat 2x2 spire caps)
    place_egg(3, 135, 233, mask=egg_flat_mask & world["nest_terrace_a"], radius=6, min_sep=6)
    place_egg(3, 164, 233, mask=egg_flat_mask & world["nest_terrace_b"], radius=6, min_sep=6)
    place_egg(3, 168, 215, mask=egg_flat_mask & world["nest_terrace_c"], radius=6, min_sep=6)
    place_egg(3, 168, 222, mask=spire_flat_mask & world["nest_spire_1"], radius=4, min_sep=6)
    place_egg(3, 147, 235, mask=spire_flat_mask & world["nest_spire_2"], radius=4, min_sep=6)

    # Species 4 (5 eggs in Ancient Nest Lagoon on lagoon islands & shores, strictly on flat_3x3_mask)
    place_egg(4, 143, 218, mask=egg_flat_mask & world["lagoon_island_w"], radius=5, min_sep=6)
    place_egg(4, 155, 217, mask=egg_flat_mask & world["lagoon_island_e"], radius=5, min_sep=6)
    place_egg(4, 139, 211, min_sep=8)
    place_egg(4, 156, 211, min_sep=8)
    place_egg(4, 160, 227, min_sep=8)

    # ---------------- GasVents ----------------
    gv_spots = [
        (147, 126, (kind == KIND_FLOOR) & (zone == ZONE_LAVA)),
        (162, 153, (kind == KIND_FLOOR) & (zone == ZONE_LAVA)),
        (164, 224, dry_mask & (zone == ZONE_LAGOON)),
        (143, 235, dry_mask & (zone == ZONE_LAGOON)),
        (153, 237, dry_mask & (zone == ZONE_LAGOON)),
    ]
    for gc, gr, gmask in gv_spots:
        c, r = nearest_dry_cell(P, world, gc, gr, allowed_mask=gmask, radius=8)
        P.add(c, r, IT_GASVENT, (0, 0, 0, 0), "special", protect=True)

    # ---------------- RollBoulders on hm=255 canyon rims ----------------
    # Place along the Lava->Canyon defile & Zone 3 canyon rims
    rim_mask = (
        (kind == KIND_CLIFF)
        & dilate_n((zone == ZONE_CANYON) & walk, 3, 8)
        & ~dilate_n(walk, 2, 8)
    )
    defile_rim_mask = (
        rim_mask
        & (np.arange(D)[:, None] >= 156)
        & (np.arange(D)[:, None] <= 184)
        & (np.arange(W)[None, :] >= 138)
        & (np.arange(W)[None, :] <= 176)
    )
    rb_placed = []
    defile_cells = [(int(r), int(c)) for r, c in np.argwhere(defile_rim_mask)]
    rng.shuffle(defile_cells)
    for r, c in defile_cells:
        if len(rb_placed) >= 4:
            break
        if any((r - pr) ** 2 + (c - pc) ** 2 < 8 ** 2 for pr, pc in rb_placed):
            continue
        if 8 <= r < D - 8 and 8 <= c < W - 8:
            P.add(c, r, IT_ROLLBOULDER, (0, 0, 0, 0), "scenery", protect=False)
            rb_placed.append((r, c))

    rim_cells = [(int(r), int(c)) for r, c in np.argwhere(rim_mask)]
    rng.shuffle(rim_cells)
    for r, c in rim_cells:
        if len(rb_placed) >= 8:
            break
        if any((r - pr) ** 2 + (c - pc) ** 2 < 9 ** 2 for pr, pc in rb_placed):
            continue
        if 8 <= r < D - 8 and 8 <= c < W - 8:
            P.add(c, r, IT_ROLLBOULDER, (0, 0, 0, 0), "scenery", protect=False)
            rb_placed.append((r, c))

    # ---------------- Crystal Bowl (~24 Crystals parm0=0..2 in Zone 3) ----------------
    bowl_cells = [
        (int(r), int(c))
        for r, c in np.argwhere(world["crystal_bowl"] & dry_mask)
        if (r, c) in bfs
    ]
    P.scatter(24, bowl_cells, IT_CRYSTAL, lambda i: (i % 3, 0, 0, 0), "scenery", 2.0)

    # ---------------- Bushes (parm3=1 Tricer ambushes) ----------------
    P.scatter(5, zone_flat[ZONE_LAVA], IT_BUSH, lambda i: (0, 0, 0, 1 if i < 4 else 0), "scenery", 4.0)
    P.scatter(5, zone_flat[ZONE_CANYON], IT_BUSH, lambda i: (0, 0, 0, 1 if i < 3 else 0), "scenery", 4.0)

    # ---------------- Secret Alcove PowerUp ----------------
    alc_c, alc_r = nearest_dry_cell(
        P, world, 158, 191,
        allowed_mask=flat_3x3_mask & world["secret_alcove"], radius=5,
    )
    P.add(alc_c, alc_r, IT_POWERUP, (POW_NUKE, 0, 0, 0), "powerup", protect=False)

    # ---------------- PowerUps (1/4-scale of original 86 -> ~25 normal, ~77 Pro) ----------------
    pow_plan = [
        (ZONE_MARSH, [POW_HEALTH, POW_LASER, POW_HEAT, POW_TRI, POW_SHIELD]),
        (ZONE_LAVA, [POW_HEALTH, POW_SHIELD, POW_LASER, POW_HEAT, POW_TRI, POW_SONIC]),
        (ZONE_CANYON, [POW_HEALTH, POW_LASER, POW_TRI, POW_SONIC, POW_HEAT]),
        (ZONE_LAGOON, [POW_HEALTH, POW_SHIELD, POW_LASER, POW_TRI, POW_HEAT, POW_SONIC, POW_NUKE]),
    ]
    for zi, plist in pow_plan:
        lst = list(plist)
        rng.shuffle(lst)
        got = P.scatter(len(lst), zone_dry[zi], IT_POWERUP, lambda i, l=lst: (l[i], 0, 0, 0), "powerup", 5.5)
        if got < len(lst):
            raise SystemExit(f"powerup placement failed in zone {zi} ({got}/{len(lst)})")

    pro_pows = (
        [POW_TRI] * 14 + [POW_LASER] * 14 + [POW_NUKE] * 8
        + [POW_HEAT] * 7 + [POW_HEALTH] * 5 + [POW_SHIELD] * 4
    )
    rng.shuffle(pro_pows)
    for i, pw in enumerate(pro_pows):
        zi = i % 4
        P.scatter(1, zone_dry[zi], IT_POWERUP, lambda _n, p=pw: (p, 0, 0, 0), "powerup", 5.0, pro_only=True)

    # ---------------- Enemies (1/4-scale of original 571 -> ~150 normal; Pro ~320) ----------------
    enemy_plan = [
        (ZONE_MARSH, [(IT_REX, 10), (IT_STEGO, 12)]),
        (ZONE_LAVA, [(IT_PTERA, 12), (IT_TRICER, 8), (IT_REX, 14)]),
        (ZONE_CANYON, [(IT_SPITTER, 14), (IT_REX, 12), (IT_TRICER, 7), (IT_PTERA, 7)]),
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
    P.scatter(42, zone_flat[ZONE_MARSH], IT_TREE,
              lambda _i: (rng.choice((TREE_FERN, TREE_BAMBOO, TREE_CYPRESS, TREE_STICKPALM)), 0, 0, 0),
              "tree", 2.5)
    P.scatter(10, zone_flat[ZONE_MARSH], IT_MUSHROOM, lambda _n: (0, 0, 0, 0), "scenery", 2.8)
    P.scatter(5, zone_flat[ZONE_MARSH], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)
    P.scatter(8, zone_flat[ZONE_MARSH], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)

    P.scatter(18, zone_flat[ZONE_LAVA], IT_TREE,
              lambda _i: (rng.choice((TREE_FERN, TREE_PINEPALM)), 0, 0, 0),
              "tree", 3.0)
    P.scatter(6, zone_flat[ZONE_LAVA], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)

    oasis_shore = [
        (int(r), int(c))
        for r, c in np.argwhere(
            dilate8(dilate8(is_water_bed)) & (zone == ZONE_CANYON) & (kind == KIND_FLOOR)
        )
        if (r, c) in bfs
    ]
    P.scatter(14, oasis_shore, IT_TREE,
              lambda _i: (rng.choice((TREE_MAINPALM, TREE_STICKPALM)), 0, 0, 0), "tree", 2.5)
    P.scatter(22, zone_flat[ZONE_CANYON], IT_TREE,
              lambda _i: (rng.choice((TREE_STICKPALM, TREE_MAINPALM, TREE_FERN)), 0, 0, 0),
              "tree", 2.8)
    P.scatter(12, zone_flat[ZONE_CANYON], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 2.8)
    P.scatter(5, zone_flat[ZONE_CANYON], IT_BOULDER, lambda _n: (0, 0, 0, 0), "scenery", 3.5)

    P.scatter(34, zone_flat[ZONE_LAGOON], IT_TREE,
              lambda _i: (rng.choice((TREE_BAMBOO, TREE_MAINPALM, TREE_STICKPALM, TREE_FERN)), 0, 0, 0),
              "tree", 2.6)
    P.scatter(8, zone_flat[ZONE_LAGOON], IT_SPOREPOD, lambda _n: (0, 0, 0, 0), "scenery", 3.0)

    return P, bfs, (sc, sr), bridge_points
