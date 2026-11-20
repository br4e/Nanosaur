"""World layout ("Pangaea Basin"), vertex heightfield, zone basins, and landmarks."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from .constants import (
    D,
    KIND_CLIFF,
    KIND_CRAMP_L,
    KIND_CRAMP_U,
    KIND_FLOOR,
    KIND_FORD,
    KIND_LAVA_BED,
    KIND_SPIRE_178,
    KIND_TERRACE_109,
    KIND_TERRACE_128,
    KIND_WALK_RAMP,
    KIND_WATER_BED,
    KIND_WATER_SHELF,
    W,
    ZONE_CANYON,
    ZONE_LAGOON,
    ZONE_LAVA,
    ZONE_MARSH,
)
from .geometry import (
    dilate8,
    dilate_n,
    ensure_smooth_walkable_heightfield,
    paint_blob,
    paint_capsule,
    smooth_noise_field,
)


def _rect_mask(r0, r1, c0, c1):
    m = np.zeros((D, W), dtype=bool)
    m[r0:r1, c0:c1] = True
    return m


def build_world(rng):
    """Build the 4-zone Pangaea Basin layout, vertex heightfield V, and region masks."""
    del rng  # deterministic geometric layout
    cc, rr = np.meshgrid(np.arange(W), np.arange(D))
    noise = smooth_noise_field(cc, rr)

    # ---- Zone 1: Emerald Marsh (NW) ----
    # Includes a grassy NW overlook spur around (88, 121) for Portal P0
    z_marsh = (
        paint_blob(cc, rr, 102, 136, 20.5, 19.0, -15, warp=0.14 * noise)
        | paint_blob(cc, rr, 92, 152, 12.5, 13.0, 20, warp=0.12 * noise)
        | paint_blob(cc, rr, 114, 124, 12.5, 11.0, 10, warp=0.10 * noise)
        | paint_blob(cc, rr, 91, 122, 8.2, 7.2, -10, warp=0.06 * noise)
    )

    # ---- Zone 2: Obsidian Lava Gorge (NE) ----
    z_lava = (
        paint_blob(cc, rr, 159, 137, 21.0, 17.5, 12, warp=0.12 * noise)
        | paint_blob(cc, rr, 143, 124, 11.5, 10.0, -18, warp=0.10 * noise)
        | paint_blob(cc, rr, 173, 153, 11.5, 11.5, 25, warp=0.10 * noise)
    )

    # ---- Zone 3: Crystal Canyon & Oasis Crossroads (Central/W) ----
    # Includes:
    #  - winding narrow Lava->Canyon defile (3-5 tiles wide, hw=2.0)
    #  - Crystal Bowl + Oasis Crossroads + P2 overlook spur around (121, 169)
    #  - secret hidden side alcove behind a narrow 3-4 tile throat around (145..161, 187..196)
    #  - narrow Western return canyon (4-5 tiles wide, hw=2.2)
    z_canyon = (
        paint_capsule(cc, rr, (169, 160), (154, 170), 2.1, 2.2, warp=0.30 * noise)
        | paint_capsule(cc, rr, (154, 170), (144, 179), 2.3, 4.2, warp=0.35 * noise)
        | paint_blob(cc, rr, 132, 181, 14.2, 11.2, -15, warp=0.11 * noise)   # Crystal Bowl + Crossroads
        | paint_blob(cc, rr, 119, 172, 7.8, 7.2, 5, warp=0.06 * noise)       # P2 Overlook spur
        | paint_blob(cc, rr, 108, 189, 12.0, 10.5, 10, warp=0.10 * noise)    # Palm Oasis
        | paint_capsule(cc, rr, (98, 184), (92, 163), 2.2, 2.3, warp=0.25 * noise)  # Western return canyon
        # Secret side alcove east of Crystal Bowl (narrow 3-4 tile throat -> 7-tile chamber)
        | paint_capsule(cc, rr, (143, 185), (151, 188), 1.8, 1.8, warp=0.18 * noise)
        | paint_blob(cc, rr, 156, 190, 4.3, 4.1, 10, warp=0.08 * noise)
    )

    # ---- Zone 4: Ancient Nest Lagoon (South Endgame) ----
    z_lagoon = (
        paint_blob(cc, rr, 148, 226, 21.8, 18.0, -8, warp=0.12 * noise)
        | paint_blob(cc, rr, 168, 218, 10.5, 10.0, 20, warp=0.10 * noise)
        | paint_blob(cc, rr, 126, 215, 7.2, 6.8, -5, warp=0.06 * noise)      # P3 Overlook terrace bay
    )

    # ---- Narrow Dramatic S-curve Connectors ----
    # 1) Marsh -> Lava Gorge throat (~4 tiles wide, hw ~ 2.0)
    conn_marsh_lava = (
        paint_capsule(cc, rr, (122, 127), (130, 129), 2.1, 2.0, warp=0.25 * noise)
        | paint_capsule(cc, rr, (130, 129), (138, 126), 2.0, 2.2, warp=0.25 * noise)
    )
    # 2) Lava Gorge -> Crystal Canyon winding defile throat (~3-4 tiles wide)
    conn_lava_canyon = paint_capsule(cc, rr, (171, 154), (166, 163), 2.1, 2.0, warp=0.25 * noise)
    # 3) Crystal Canyon -> Nest Lagoon gate throat (~4 tiles wide) opening onto P3 terrace
    conn_canyon_lagoon = (
        paint_capsule(cc, rr, (124, 194), (125, 202), 2.1, 2.0, warp=0.20 * noise)
        | paint_capsule(cc, rr, (125, 202), (127, 210), 2.0, 2.3, warp=0.20 * noise)
    )

    walk = (
        z_marsh | z_lava | z_canyon | z_lagoon
        | conn_marsh_lava | conn_lava_canyon | conn_canyon_lagoon
    )

    # Carve rocky mesa pillars/peninsulas so connectors & alcove stay distinct and dramatic
    rock_pillars = (
        paint_blob(cc, rr, 128, 158, 8.0, 9.2, 8)     # central ridge dividing Marsh/Lava from Canyon
        | paint_blob(cc, rr, 106, 169, 6.2, 7.2, -10) # western canyon bend divider
        | paint_blob(cc, rr, 141, 201, 7.2, 5.2, 10)  # ridge isolating secret alcove from Lagoon north
        | paint_blob(cc, rr, 156, 178, 4.8, 3.8, -15) # ridge shaping Lava->Canyon defile rim
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

    # Zone map with smooth organic noise-warped boundaries
    warp_c = cc + 4.0 * noise
    warp_r = rr + 4.0 * np.roll(noise, 7, axis=0)
    d_m = (warp_c - 102) ** 2 + (warp_r - 137) ** 2
    d_l = (warp_c - 159) ** 2 + (warp_r - 137) ** 2
    d_c = (warp_c - 122) ** 2 + (warp_r - 183) ** 2
    d_g = (warp_c - 148) ** 2 + (warp_r - 224) ** 2
    zone = np.argmin(np.stack([d_m, d_l, d_c, d_g]), axis=0).astype(np.uint8)
    zone[z_marsh] = ZONE_MARSH
    zone[z_lava] = ZONE_LAVA
    zone[z_canyon] = ZONE_CANYON
    zone[z_lagoon] = ZONE_LAGOON

    marsh_lava_split = 130.0 + 2.5 * noise
    zone[conn_marsh_lava & (cc < marsh_lava_split)] = ZONE_MARSH
    zone[conn_marsh_lava & (cc >= marsh_lava_split)] = ZONE_LAVA

    lava_canyon_split = 158.0 + 2.0 * noise
    zone[conn_lava_canyon & (rr < lava_canyon_split)] = ZONE_LAVA
    zone[conn_lava_canyon & (rr >= lava_canyon_split)] = ZONE_CANYON

    canyon_lagoon_split = 203.0 + 2.0 * noise
    zone[conn_canyon_lagoon & (rr < canyon_lagoon_split)] = ZONE_CANYON
    zone[conn_canyon_lagoon & (rr >= canyon_lagoon_split)] = ZONE_LAGOON
    # Keep secret alcove strictly in Canyon
    zone[(rr >= 180) & (rr <= 198) & (cc >= 142) & (cc <= 165)] = ZONE_CANYON

    # Cliff bands around walkable basin
    kind = np.full((D, W), KIND_CLIFF, dtype=np.uint8)
    kind[walk] = KIND_FLOOR

    d1 = dilate8(walk) & ~walk
    d2 = dilate8(dilate8(walk)) & ~walk & ~d1
    kind[d1] = KIND_CRAMP_L
    kind[d2] = KIND_CRAMP_U

    cliff_near2 = dilate_n(~walk, 2, 8)
    cliff_near3 = dilate_n(~walk, 3, 8)
    cliff_near4 = dilate_n(~walk, 4, 8)
    cliff_near5 = dilate_n(~walk, 5, 8)

    # =====================================================================
    # ZONE FEATURES (Water lakes, Lava basin/islands, Overlooks, Terraces)
    # =====================================================================
    V = np.full((D + 1, W + 1), 255, dtype=np.int32)

    # --- 1. Emerald Marsh Lakes (3 organic lakes, hm=41) + shallow ford island ---
    marsh_lake_1 = (
        paint_blob(cc, rr, 105, 141, 9.0, 7.6, 12, warp=0.14 * noise)
        & walk & ~cliff_near2
    )
    marsh_island = paint_blob(cc, rr, 105, 141, 3.0, 2.8, 15, warp=0.10 * noise) & marsh_lake_1
    marsh_ford = (
        paint_capsule(cc, rr, (96, 141), (103, 141), 1.45, 1.5, warp=0.30 * noise)
        & marsh_lake_1 & ~marsh_island
    )
    marsh_lake_1_water = marsh_lake_1 & ~marsh_island & ~marsh_ford

    marsh_lake_2 = (
        paint_blob(cc, rr, 90, 152, 4.2, 3.8, -20, warp=0.12 * noise)
        & walk & ~cliff_near2
    )
    marsh_lake_3 = (
        paint_blob(cc, rr, 115, 124, 3.8, 3.3, 15, warp=0.10 * noise)
        & walk & ~cliff_near2
    )

    # --- 3b. Palm Oasis in Zone 3 (around 108, 189) ---
    oasis_pool = (
        paint_blob(cc, rr, 108, 189, 4.6, 4.0, -12, warp=0.12 * noise)
        & walk & ~cliff_near2
    )

    # --- 4. Ancient Nest Lagoon Lake (around 149, 218) with 2 lush islands ---
    lagoon_lake = (
        paint_blob(cc, rr, 149, 218, 11.2, 5.0, -5, warp=0.10 * noise)
        & walk & ~cliff_near2
    )
    lagoon_island_w = paint_blob(cc, rr, 143, 218, 2.8, 2.5, -15, warp=0.08 * noise) & lagoon_lake
    lagoon_island_e = paint_blob(cc, rr, 155, 217, 2.8, 2.5, 18, warp=0.08 * noise) & lagoon_lake
    lagoon_ford_w = (
        paint_capsule(cc, rr, (137, 218), (141, 218), 1.4, 1.4, warp=0.25 * noise)
        & lagoon_lake & ~lagoon_island_w
    )
    lagoon_ford_e = (
        paint_capsule(cc, rr, (157, 217), (161, 217), 1.4, 1.4, warp=0.25 * noise)
        & lagoon_lake & ~lagoon_island_e
    )
    lagoon_water = (
        lagoon_lake & ~lagoon_island_w & ~lagoon_island_e & ~lagoon_ford_w & ~lagoon_ford_e
    )

    all_water_beds = marsh_lake_1_water | marsh_lake_2 | marsh_lake_3 | oasis_pool | lagoon_water
    all_fords = marsh_ford | lagoon_ford_w | lagoon_ford_e
    all_water_islands = marsh_island | lagoon_island_w | lagoon_island_e

    # --- 2. Obsidian Lava Gorge (NE): wide organic lava river/basin + Portal Island ---
    lava_basin = (
        paint_blob(cc, rr, 159, 138, 17.2, 12.6, 10, warp=0.13 * noise)
        & z_lava & walk & ~cliff_near2
    )
    # P1 Portal Island: guaranteed >= 5x5 flat top centered at (c=156, r=138)
    lava_portal_island = (
        (paint_blob(cc, rr, 156, 138, 4.1, 3.7, -10, warp=0.08 * noise) | _rect_mask(136, 141, 154, 159))
        & walk & ~cliff_near2
    )
    lava_north_island = (
        (paint_blob(cc, rr, 155, 125, 3.4, 3.0, 15, warp=0.10 * noise) | _rect_mask(124, 127, 154, 157))
        & walk & ~cliff_near2
    )
    lava_east_island = (
        (paint_blob(cc, rr, 174, 141, 3.4, 3.0, -15, warp=0.10 * noise) | _rect_mask(140, 143, 173, 176))
        & walk & ~cliff_near2
    )

    lava_islands = lava_portal_island | lava_north_island | lava_east_island
    lava_bed = lava_basin & ~lava_islands
    bridge_a_channel = (
        paint_capsule(cc, rr, (141, 138), (152, 138), 3.0, 3.0, warp=0.3 * noise)
        & walk & ~lava_islands
    )
    bridge_b_channel = (
        paint_capsule(cc, rr, (160, 140), (171, 141), 3.0, 3.0, warp=0.3 * noise)
        & walk & ~lava_islands
    )
    lava_bed |= (bridge_a_channel | bridge_b_channel)
    lava_bed &= ~lava_islands

    # --- 3. Crystal Bowl (around 132, 181) ---
    crystal_bowl = (
        paint_blob(cc, rr, 132, 181, 8.0, 6.0, -15, warp=0.05 * noise)
        & walk & ~cliff_near2
    )

    # --- Dedicated Raised Portal Overlook Plateaus (>= 5x5 flat top, smooth ramp) ---
    # P0 (Marsh): raised hm=109 grassy overlook plateau at (c=92, r=122) overlooking Marsh Lake 1
    marsh_p0_c, marsh_p0_r = 92, 122
    marsh_p0_terrace = _rect_mask(marsh_p0_r - 2, marsh_p0_r + 3, marsh_p0_c - 2, marsh_p0_c + 3) & walk & ~cliff_near3

    # P1 (Lava Gorge): centered at (c=156, r=138) on flat hm=85 lava_portal_island
    lava_p1_c, lava_p1_r = 156, 138

    # P2 (Crystal Canyon): raised hm=109 overlook terrace at (c=119, r=172) overlooking Crystal Bowl & Oasis
    canyon_p2_c, canyon_p2_r = 119, 172
    canyon_p2_terrace = _rect_mask(canyon_p2_r - 2, canyon_p2_r + 3, canyon_p2_c - 2, canyon_p2_c + 3) & walk & ~cliff_near3

    # P3 (Nest Lagoon): raised hm=109 overlook terrace at (c=127, r=215) overlooking the endgame lagoon
    lagoon_p3_c, lagoon_p3_r = 127, 215
    lagoon_p3_terrace = _rect_mask(213, 218, 125, 130) & walk & ~cliff_near3

    # --- Raised Basalt Overlook (hm=109) in Zone 2 & Nest Terraces/Spires in Zone 4 ---
    lava_overlook = _rect_mask(120, 124, 141, 145) & walk & ~lava_bed & ~cliff_near3
    nest_terrace_a = _rect_mask(231, 235, 133, 137) & walk & ~lagoon_lake & ~cliff_near3
    nest_terrace_b = _rect_mask(231, 235, 162, 166) & walk & ~lagoon_lake & ~cliff_near3
    nest_terrace_c = _rect_mask(213, 217, 166, 170) & walk & ~lagoon_lake & ~cliff_near3

    # Two raised spires (hm=178) with strictly flat 2x2 cap tiles (3x3 vertices)
    lake_near4 = dilate_n(lagoon_lake, 4, 8)
    lake_near5 = dilate_n(lagoon_lake, 5, 8)
    nest_spire_1 = _rect_mask(222, 224, 168, 170) & walk & ~lake_near5 & ~cliff_near4
    nest_spire_2 = _rect_mask(234, 236, 146, 148) & walk & ~lake_near4 & ~cliff_near5

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
    water_buffer = dilate_n(all_water_beds | all_fords | all_water_islands, 4, 8)
    lava_buffer = dilate8(lava_bed | lava_islands)
    ramp_domain = walk & ~water_buffer & ~lava_buffer & ~dilate8(~walk)

    # Carve raised overlooks and terraces (hm=109, 128) with gentle ramps
    carve_vertex_ramp(marsh_p0_terrace, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(canyon_p2_terrace, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(lagoon_p3_terrace, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(lava_overlook, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(nest_terrace_a, 109, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(nest_terrace_b, 128, ramp_domain, base_h=85, max_step=15)
    carve_vertex_ramp(nest_terrace_c, 128, ramp_domain, base_h=85, max_step=15)

    # Carve spires (hm=178) with smooth 7-step skirts (max_step=14 -> <=14px per tile)
    carve_vertex_ramp(nest_spire_1, 178, ramp_domain, base_h=85, max_step=14)
    carve_vertex_ramp(nest_spire_2, 178, ramp_domain, base_h=85, max_step=14)

    # 2. Water basins in vertex space
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

    # 3. Lava basin in vertex space
    v_lava = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(lava_bed)):
        v_lava[r:r + 2, c:c + 2] = True
    v_lava_isl = np.zeros((D + 1, W + 1), dtype=bool)
    for r, c in zip(*np.nonzero(lava_islands)):
        v_lava_isl[r:r + 2, c:c + 2] = True
    v_lava &= walk_verts & ~cliff_v & ~v_lava_isl
    V[v_lava] = 67
    V[v_lava_isl & walk_verts] = 85

    # Lock cliff boundary vertices & feature plateaus, then relax walkable steps <= 20px
    v_locked = cliff_v | (V <= 67) | v_lava_isl
    for m in (
        marsh_island, lagoon_island_w, lagoon_island_e,
        marsh_p0_terrace, canyon_p2_terrace, lagoon_p3_terrace,
        lava_overlook, nest_terrace_a, nest_terrace_b, nest_terrace_c,
        nest_spire_1, nest_spire_2,
    ):
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
    is_lava_rim = walk & (zone == ZONE_LAVA) & (tmin == 67) & (tmax > 67)

    # Classify terraces & ramps
    kind[is_water_bed] = KIND_WATER_BED
    kind[is_water_47 | is_water_shore] = KIND_WATER_SHELF
    kind[is_ford_tile] = KIND_FORD
    kind[is_lava_tile] = KIND_LAVA_BED
    kind[walk & (tmin == 109) & (tmax == 109)] = KIND_TERRACE_109
    kind[walk & (tmin == 128) & (tmax == 128)] = KIND_TERRACE_128
    kind[walk & (tmin == 178) & (tmax == 178)] = KIND_SPIRE_178
    kind[walk & (kind == KIND_FLOOR) & ((tmin != 85) | (tmax != 85))] = KIND_WALK_RAMP

    # Lagoon grass meadows (lush islands + one cohesive southern meadow)
    lagoon_meadow = (
        (zone == ZONE_LAGOON)
        & walk
        & ~is_water_bed & ~is_water_47 & ~is_water_shore & ~is_ford_tile
        & (kind != KIND_TERRACE_109) & (kind != KIND_TERRACE_128)
        & (kind != KIND_SPIRE_178) & (kind != KIND_WALK_RAMP)
        & (
            lagoon_island_w | lagoon_island_e
            | paint_blob(cc, rr, 152, 232, 11.0, 7.5, -5, warp=0.05 * noise)
        )
    )
    lagoon_meadow &= ~dilate_n(lagoon_p3_terrace, 3, 8)
    lagoon_meadow &= ~dilate8(nest_terrace_a | nest_terrace_b | nest_terrace_c | nest_spire_1 | nest_spire_2)

    # Portal locations (guaranteed 3x3 flat, path==0)
    portal_coords = {
        0: (marsh_p0_c, marsh_p0_r),
        1: (lava_p1_c, lava_p1_r),
        2: (canyon_p2_c, canyon_p2_r),
        3: (lagoon_p3_c, lagoon_p3_r),
    }

    # Secret alcove chamber mask (for dedicated Species 2 egg + powerup)
    secret_alcove = paint_blob(cc, rr, 156, 190, 3.6, 3.4, 10) & walk

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
        "lagoon_meadow": lagoon_meadow,
        "lava_portal_island": lava_portal_island,
        "lava_north_island": lava_north_island,
        "lava_east_island": lava_east_island,
        "marsh_p0_terrace": marsh_p0_terrace,
        "canyon_p2_terrace": canyon_p2_terrace,
        "lagoon_p3_terrace": lagoon_p3_terrace,
        "lava_overlook": lava_overlook,
        "crystal_bowl": crystal_bowl,
        "secret_alcove": secret_alcove,
        "nest_terrace_a": nest_terrace_a,
        "nest_terrace_b": nest_terrace_b,
        "nest_terrace_c": nest_terrace_c,
        "nest_spire_1": nest_spire_1,
        "nest_spire_2": nest_spire_2,
        "portal_coords": portal_coords,
        "rock_pillars": rock_pillars,
    }
