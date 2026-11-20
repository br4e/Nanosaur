"""Strict 11-gate validation suite for generated .ter and Map.tga outputs."""

from __future__ import annotations

import math
import struct
from collections import Counter, deque

import numpy as np

from .constants import (
    D,
    ENEMY_TYPES,
    IT_CRYSTAL,
    IT_EGG,
    IT_LAVA,
    IT_MUSHROOM,
    IT_PORTAL,
    IT_POWERUP,
    IT_ROLLBOULDER,
    IT_START,
    IT_STEPSTONE,
    IT_WATER,
    PATH_OPEN,
    PATH_SOLID_ALL,
    TILE_ATTRIB_LAVA,
    TILE_ATTRIB_WATER,
    W,
)


def _chebyshev_dist_map(seed_mask):
    """Exact Chebyshev (8-connected) distance in tiles from `seed_mask`."""
    d_rows, d_cols = seed_mask.shape
    dist = np.full((d_rows, d_cols), 999, dtype=np.int32)
    dq = deque()
    for r, c in np.argwhere(seed_mask):
        dist[r, c] = 0
        dq.append((int(r), int(c)))
    while dq:
        r, c = dq.popleft()
        nd = dist[r, c] + 1
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < d_rows and 0 <= nc < d_cols and dist[nr, nc] > nd:
                    dist[nr, nc] = nd
                    dq.append((nr, nc))
    return dist


def validate_ter(path, orig, label):
    """Run all terrain checks (1..8, 10..11) on `path`; return (checks, stats, items)."""
    del label
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
    attribs = [
        struct.unpack_from(">Hhbbh", data, attrib_off + i * 8)
        for i in range((anim_off - attrib_off) // 8)
    ]
    num_items = struct.unpack_from(">i", data, obj_off)[0]
    items = [
        struct.unpack_from(">hhh4bHii", data, obj_off + 4 + i * 20)
        for i in range(num_items)
    ]

    # Check 1: Header & sizes
    checks.append((
        "1 header/sizes",
        ok and n_hm <= 300 and int((tex & 0xFFF).max()) < 258,
        f"{w}x{d}, {n_hm} hm tiles (<=300), max tex {(tex & 0xFFF).max()}, {num_items} items",
    ))

    # Check 2: Item rules
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
    checks.append((
        "2 item rules",
        sorted_ok and len(starts) == 1 and eggs_ok
        and portals == [0, 1, 2, 3] and rollboulder_count >= 4 and types_ok and border_ok,
        f"sorted={sorted_ok} starts={len(starts)} eggs/sp={dict(egg_by_sp)} "
        f"portals={portals} rollboulders={rollboulder_count}",
    ))

    # Reconstruct exact vertex heightfield V_read[r,c] = corner[0,0] post-flip
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
    edge_touch = bool(
        np.any(walk_open[0, :]) or np.any(walk_open[-1, :])
        or np.any(walk_open[:, 0]) or np.any(walk_open[:, -1])
    )
    bfs_count = int(seen.sum())
    checks.append((
        "3 containment/BFS",
        cont_ok and not edge_touch and 5800 <= bfs_count <= 7000,
        f"border_solid={cont_ok}, edge_touch={edge_touch}, bfs_tiles={bfs_count}",
    ))

    # Check 4: Connectivity of all 25 eggs, 4 portals, all powerups, crystals, mushrooms, enemies
    check_types = {IT_EGG, IT_PORTAL, IT_POWERUP, IT_CRYSTAL, IT_MUSHROOM} | set(ENEMY_TYPES)
    unreachable = [
        (it[2], it[0] // 32, it[1] // 32)
        for it in items
        if it[2] in check_types and not seen[it[1] // 32, it[0] // 32]
    ]
    checks.append((
        "4 connectivity",
        not unreachable,
        f"unreachable={unreachable[:6]} (25 eggs + 4 portals + all powerups/enemies reachable)",
    ))

    # Check 5: Exact 8x8 Water & Lava coverage + zero dry leaks
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

    checks.append((
        "5 water/lava",
        dry_water_leaks == 0 and dry_lava_leaks == 0 and uncov_water == 0 and uncov_lava == 0,
        f"uncov_water={uncov_water}, uncov_lava={uncov_lava}, "
        f"dry_water_leaks={dry_water_leaks}, dry_lava_leaks={dry_lava_leaks}",
    ))

    # Check 6: Vertex continuity & smooth walkable steps
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
                if (
                    int(t[0, 31]) != int(V_read[r, c + 1])
                    or int(t[31, 0]) != int(V_read[r + 1, c])
                    or int(t[31, 31]) != int(V_read[r + 1, c + 1])
                ):
                    corner_mismatch += 1
                for dr, dc in ((0, 0), (0, 1), (1, 0), (1, 1)):
                    if not walk_p0[r + dr, c + dc] and int(V_read[r + dr, c + dc]) != 85:
                        perimeter_spikes += 1
            elif walk_p0[r, c + 1] or walk_p0[r + 1, c] or walk_p0[r + 1, c + 1] or walk_p0[r - 1, c] or walk_p0[r, c - 1]:
                if walk_p0[r - 1, c - 1] or walk_p0[r - 1, c] or walk_p0[r, c - 1]:
                    if int(V_read[r, c]) != 85:
                        perimeter_spikes += 1

    checks.append((
        "6 vertex continuity",
        has_182_walk == 0 and corner_mismatch == 0 and perimeter_spikes == 0
        and max_step_nonspire <= 20 and max_step_all <= 22,
        f"182_on_path0={has_182_walk}, corner_mismatch={corner_mismatch}, "
        f"perimeter_spikes={perimeter_spikes}, max_step_nonspire={max_step_nonspire}, "
        f"max_step_all={max_step_all}",
    ))

    # Check 7: StepStone spacing <= 3.55 tiles & on flat hm=67 lava
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
    checks.append((
        "7 stepstones",
        stone_ok,
        f"count={len(stones)}, all_reincarnate={all(s[2] == 1 for s in stones)}, max_nn={max_stone_nn:.2f} tiles",
    ))

    # Check 8: Spacing (no duplicate item tile coordinates)
    tiles_used = Counter((it[0] // 32, it[1] // 32) for it in items)
    dup = [k for k, v in tiles_used.items() if v > 1]
    checks.append(("8 item spacing", not dup, f"dup_tiles={len(dup)}"))

    # Check 10: Flat Portals & Eggs (strict 3x3 flat on portals & non-spire eggs; strictly flat on spires)
    dist_cliff = _chebyshev_dist_map(pathl != PATH_OPEN)
    dist_water = _chebyshev_dist_map(w_attr)
    dist_lava = _chebyshev_dist_map(l_attr)

    def _is_3x3_flat(r, c):
        if r < 1 or r >= d - 2 or c < 1 or c >= w - 2:
            return False
        h0 = int(tile_min[r, c])
        return bool(
            np.all(pathl[r - 1:r + 2, c - 1:c + 2] == PATH_OPEN)
            and np.all(V_read[r - 1:r + 3, c - 1:c + 3] == h0)
            and np.all(tile_min[r - 1:r + 2, c - 1:c + 2] == h0)
            and np.all(tile_max[r - 1:r + 2, c - 1:c + 2] == h0)
        )

    portal_issues = []
    for it in items:
        if it[2] != IT_PORTAL:
            continue
        pid = it[3]
        c, r = it[0] // 32, it[1] // 32
        min_lava = 2 if pid == 1 else 3
        ok_p = (
            _is_3x3_flat(r, c)
            and dist_cliff[r, c] >= 3
            and dist_water[r, c] >= 3
            and dist_lava[r, c] >= min_lava
        )
        if not ok_p:
            portal_issues.append((pid, c, r, int(dist_cliff[r, c]), int(dist_water[r, c]), int(dist_lava[r, c])))

    egg_issues = []
    spire_eggs = 0
    for it in items:
        if it[2] != IT_EGG:
            continue
        c, r = it[0] // 32, it[1] // 32
        is_spire = (tile_min[r, c] == 178 and tile_max[r, c] == 178)
        if is_spire:
            spire_eggs += 1
            if not (pathl[r, c] == PATH_OPEN and tile_min[r, c] == tile_max[r, c]):
                egg_issues.append(("spire", it[3], c, r))
        else:
            ok_e = (
                _is_3x3_flat(r, c)
                and dist_cliff[r, c] >= 2
                and dist_water[r, c] >= 2
                and dist_lava[r, c] >= 2
            )
            if not ok_e:
                egg_issues.append(("ground", it[3], c, r, int(tile_min[r, c]), int(tile_max[r, c]),
                                   int(dist_cliff[r, c]), int(dist_water[r, c]), int(dist_lava[r, c])))

    checks.append((
        "10 flat portals/eggs",
        not portal_issues and not egg_issues,
        f"portal_issues={portal_issues}, egg_issues={egg_issues}, spire_eggs={spire_eggs}",
    ))

    # Check 11: Texture Cohesion / Anti-Mosaic on path == 0
    rgb = orig.tile_avg[tex & 0x0FFF].astype(np.float32)
    d_horiz = np.linalg.norm(rgb[:, :-1] - rgb[:, 1:], axis=2)[walk_p0[:, :-1] & walk_p0[:, 1:]]
    d_vert = np.linalg.norm(rgb[:-1, :] - rgb[1:, :], axis=2)[walk_p0[:-1, :] & walk_p0[1:, :]]
    all_jumps = np.concatenate([d_horiz, d_vert])
    mean_rgb_jump = float(all_jumps.mean())
    pct_gt40 = float(100.0 * np.mean(all_jumps > 40.0))
    checks.append((
        "11 texture cohesion",
        mean_rgb_jump < 14.0 and pct_gt40 < 8.0,
        f"mean_rgb_jump={mean_rgb_jump:.2f} (<14.0), pct_gt40={pct_gt40:.2f}% (<8.0%)",
    ))

    floor_pct = 100.0 * np.count_nonzero(walk_open) / (w * d)
    cliff_pct = 100.0 * np.count_nonzero(V_read == 255) / (w * d)
    stats = dict(
        floor_pct=floor_pct,
        cliff_pct=cliff_pct,
        items=num_items,
        hm_tiles=n_hm,
        bfs_cells=bfs_count,
        mean_rgb_jump=mean_rgb_jump,
        pct_gt40=pct_gt40,
    )
    return checks, stats, items


def validate_tga(path):
    """Validate Map.tga RLE colormapped 8-bit header and orientation."""
    b = path.read_bytes()
    (
        idlen, cmtype, imtype, cmorig, cmlen, cmdepth,
        _xo, _yo, w, h, bpp, desc,
    ) = struct.unpack_from("<BBBHHBHHHHBB", b, 0)
    ok = bool(
        idlen == 0 and cmtype == 1 and imtype == 9 and cmorig == 0
        and cmlen <= 256 and cmdepth == 24 and (w, h) == (W, D)
        and bpp == 8 and (desc & 0x20)
    )
    return ok, (
        f"type={imtype} (RLE colormapped), {w}x{h}, {bpp}bpp, "
        f"{cmlen} colors, descriptor=0x{desc:02x} (top-left origin)"
    )
