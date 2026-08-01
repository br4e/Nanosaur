#!/usr/bin/env python3
"""Build a full-scale, original Nanosaur 1 campaign-style level.

The visual source is the original Level1 terrain set.  Playable zones copy
contiguous source patches (preserving their authored grass tile sequences and
flip bits), while a generated high, solid cliff field encloses every route.
The result has a distinct five-branch layout with water and lava encounters.
"""

from __future__ import annotations

import math
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "Data" / "Terrain"
ORIGINAL_DIR = ROOT / "tools" / "original_terrain"

MAP_W = 270
MAP_D = 355
TILE_PX = 32
SUPERTILE_SIZE = 5

PATH_SOLID_ALL = 9
HMTILE_FLOOR = 6
HMTILE_CLIFF_TOP = 0

ITEM_START = 0
ITEM_POWERUP = 1
ITEM_TRICER = 2
ITEM_REX = 3
ITEM_LAVA = 4
ITEM_EGG = 5
ITEM_GAS_VENT = 6
ITEM_PTERA = 7
ITEM_STEGO = 8
ITEM_PORTAL = 9
ITEM_TREE = 10
ITEM_BOULDER = 11
ITEM_MUSHROOM = 12
ITEM_BUSH = 13
ITEM_WATER = 14
ITEM_CRYSTAL = 15
ITEM_SPITTER = 16
ITEM_STEPPING_STONE = 17

POW_HEATSEEK = 0
POW_LASER = 1
POW_TRIBLAST = 2
POW_HEALTH = 3
POW_SHIELD = 4
POW_NUKE = 5
POW_SONIC = 6


@dataclass(frozen=True)
class Zone:
    name: str
    cx: int
    cz: int
    rx: int
    rz: int
    source_col: int
    source_row: int


# A central landing field with five materially different branches.
# Each source patch is an authentic contiguous Level1 grass area.
ZONES = (
    Zone("landing", 135, 166, 34, 29, 14, 87),
    Zone("mosswater", 61, 75, 30, 27, 14, 87),
    Zone("ember", 208, 76, 32, 28, 54, 87),
    Zone("fernreach", 56, 244, 31, 29, 14, 127),
    Zone("flooded", 211, 244, 33, 30, 54, 127),
    Zone("cinder", 135, 310, 29, 25, 94, 87),
)

CORRIDORS = (
    ("landing", "mosswater"),
    ("landing", "ember"),
    ("landing", "fernreach"),
    ("landing", "flooded"),
    ("landing", "cinder"),
)
CORRIDOR_HALF_WIDTH = 8


def pack_item(
    x: int,
    z: int,
    item_type: int,
    parm: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> bytes:
    return struct.pack(
        ">hhh4bHii",
        x,
        z,
        item_type,
        parm[0],
        parm[1],
        parm[2],
        parm[3],
        0,
        0,
        0,
    )


def tile_center(col: int, row: int) -> tuple[int, int]:
    return col * TILE_PX + TILE_PX // 2, row * TILE_PX + TILE_PX // 2


def point_to_segment_distance(
    x: float, z: float, ax: float, az: float, bx: float, bz: float
) -> float:
    dx = bx - ax
    dz = bz - az
    length_squared = dx * dx + dz * dz
    if length_squared == 0:
        return math.hypot(x - ax, z - az)
    t = max(0.0, min(1.0, ((x - ax) * dx + (z - az) * dz) / length_squared))
    return math.hypot(x - (ax + t * dx), z - (az + t * dz))


def zone_for_cell(col: int, row: int) -> Zone | None:
    """Return the closest room containing this cell."""
    winner: tuple[float, Zone] | None = None
    for zone in ZONES:
        normalized_distance = (
            ((col - zone.cx) / zone.rx) ** 2 + ((row - zone.cz) / zone.rz) ** 2
        )
        if normalized_distance <= 1.0 and (
            winner is None or normalized_distance < winner[0]
        ):
            winner = (normalized_distance, zone)
    return winner[1] if winner else None


def is_corridor_cell(col: int, row: int, by_name: dict[str, Zone]) -> bool:
    for from_name, to_name in CORRIDORS:
        source = by_name[from_name]
        target = by_name[to_name]
        if point_to_segment_distance(
            col, row, source.cx, source.cz, target.cx, target.cz
        ) <= CORRIDOR_HALF_WIDTH:
            return True
    return False


def make_walkable_mask() -> list[list[bool]]:
    by_name = {zone.name: zone for zone in ZONES}
    mask = [[False for _ in range(MAP_W)] for _ in range(MAP_D)]

    # The outer 6 cells must remain a solid cliff boundary.
    for row in range(6, MAP_D - 6):
        for col in range(6, MAP_W - 6):
            mask[row][col] = (
                zone_for_cell(col, row) is not None
                or is_corridor_cell(col, row, by_name)
            )
    return mask


def source_tex_value(
    tex_all: tuple[int, ...], source_width: int, col: int, row: int
) -> int:
    """Use a contiguous 40x40 original terrain pattern for the current zone."""
    zone = zone_for_cell(col, row)
    if zone is None:
        # Corridors deliberately share the landing-zone sequence.
        zone = ZONES[0]

    source_col = zone.source_col + (col - zone.cx) % 40
    source_row = zone.source_row + (row - zone.cz) % 40
    return tex_all[source_row * source_width + source_col]


def cliff_tex_value(
    cliff_tex_samples: tuple[int, ...], col: int, row: int
) -> int:
    # Samples are full texture values including authored flip/rotation bits.
    return cliff_tex_samples[(col * 37 + row * 101) % len(cliff_tex_samples)]


def nearest_walkable(
    mask: list[list[bool]], want_col: int, want_row: int
) -> tuple[int, int]:
    candidates = (
        (col, row)
        for row in range(8, MAP_D - 8)
        for col in range(8, MAP_W - 8)
        if mask[row][col]
    )
    return min(
        candidates,
        key=lambda cell: (cell[0] - want_col) ** 2 + (cell[1] - want_row) ** 2,
    )


def build_items(mask: list[list[bool]]) -> list[bytes]:
    """Build a balanced, section-based encounter list sorted by X."""
    raw: list[tuple[int, int, int, tuple[int, int, int, int]]] = []

    def add(
        col: int,
        row: int,
        item_type: int,
        parm: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        col, row = nearest_walkable(mask, col, row)
        x, z = tile_center(col, row)
        raw.append((x, z, item_type, parm))

    def zone_point(name: str, x_factor: float = 0.5, z_factor: float = 0.5) -> tuple[int, int]:
        zone = next(zone for zone in ZONES if zone.name == name)
        return (
            int(zone.cx + (x_factor - 0.5) * 2 * zone.rx * 0.55),
            int(zone.cz + (z_factor - 0.5) * 2 * zone.rz * 0.55),
        )

    # Hub / goal
    add(*zone_point("landing"), ITEM_START)
    add(*zone_point("landing", 0.50, 0.28), ITEM_PORTAL, (0, 0, 0, 0))
    # Match the original level's portal rhythm: the hub is the primary return
    # point, while three remote portals shorten the longer recovery routes.
    add(*zone_point("ember", 0.26, 0.27), ITEM_PORTAL, (1, 0, 0, 0))
    add(*zone_point("flooded", 0.26, 0.28), ITEM_PORTAL, (2, 0, 0, 0))
    add(*zone_point("cinder", 0.52, 0.25), ITEM_PORTAL, (3, 0, 0, 0))
    for x_factor, z_factor, power in (
        (0.33, 0.50, POW_HEATSEEK),
        (0.67, 0.47, POW_HEALTH),
        (0.52, 0.70, POW_LASER),
    ):
        add(*zone_point("landing", x_factor, z_factor), ITEM_POWERUP, (power, 0, 0, 0))

    # Egg 0: relatively gentle water basin.
    add(*zone_point("mosswater", 0.50, 0.50), ITEM_EGG, (0, 0, 0, 1))
    for x_factor, z_factor in ((0.34, 0.47), (0.63, 0.45), (0.49, 0.67)):
        add(*zone_point("mosswater", x_factor, z_factor), ITEM_WATER, (0, 0, 0, 1))
    add(*zone_point("mosswater", 0.70, 0.70), ITEM_POWERUP, (POW_HEALTH, 0, 0, 0))
    add(*zone_point("mosswater", 0.30, 0.72), ITEM_STEGO)

    # Egg 1: fireball-free lava shelf, guarded by a Triceratops.
    add(*zone_point("ember", 0.63, 0.50), ITEM_EGG, (1, 0, 0, 1))
    for x_factor, z_factor in ((0.32, 0.38), (0.32, 0.63), (0.51, 0.73)):
        add(*zone_point("ember", x_factor, z_factor), ITEM_LAVA, (0, 0, 0, 5))
    add(*zone_point("ember", 0.70, 0.65), ITEM_POWERUP, (POW_SHIELD, 0, 0, 0))
    add(*zone_point("ember", 0.78, 0.36), ITEM_TRICER)
    add(*zone_point("ember", 0.30, 0.72), ITEM_GAS_VENT, (0, 0, 0, 1))

    # Egg 2: a plant-heavy western route with ranged threats.
    add(*zone_point("fernreach", 0.38, 0.45), ITEM_EGG, (2, 0, 0, 1))
    add(*zone_point("fernreach", 0.69, 0.58), ITEM_POWERUP, (POW_TRIBLAST, 0, 0, 0))
    add(*zone_point("fernreach", 0.68, 0.34), ITEM_PTERA, (0, 0, 0, 2))
    add(*zone_point("fernreach", 0.73, 0.70), ITEM_STEGO)
    for x_factor, z_factor, tree_type in (
        (0.27, 0.30, 0),
        (0.30, 0.68, 3),
        (0.52, 0.29, 4),
        (0.80, 0.49, 5),
    ):
        add(*zone_point("fernreach", x_factor, z_factor), ITEM_TREE, (tree_type, 0, 0, 0))

    # Egg 3: later water section with a Rex.
    add(*zone_point("flooded", 0.66, 0.46), ITEM_EGG, (3, 0, 0, 1))
    for x_factor, z_factor in ((0.30, 0.38), (0.43, 0.57), (0.30, 0.72)):
        add(*zone_point("flooded", x_factor, z_factor), ITEM_WATER, (0, 0, 0, 1))
    add(*zone_point("flooded", 0.72, 0.69), ITEM_POWERUP, (POW_NUKE, 0, 0, 0))
    add(*zone_point("flooded", 0.76, 0.33), ITEM_REX)
    add(*zone_point("flooded", 0.54, 0.71), ITEM_CRYSTAL, (1, 0, 0, 0))

    # Egg 4: final cinder arena with a second Rex and active lava.
    add(*zone_point("cinder", 0.50, 0.64), ITEM_EGG, (4, 0, 0, 1))
    for x_factor, z_factor in ((0.31, 0.38), (0.66, 0.36), (0.48, 0.48)):
        # bit 0 auto-height, bit 1 fireballs, bit 2 half-size
        add(*zone_point("cinder", x_factor, z_factor), ITEM_LAVA, (0, 0, 0, 7))
    add(*zone_point("cinder", 0.72, 0.68), ITEM_POWERUP, (POW_SONIC, 0, 0, 0))
    add(*zone_point("cinder", 0.73, 0.31), ITEM_REX)
    add(*zone_point("cinder", 0.28, 0.70), ITEM_SPITTER)

    # Section vegetation is deliberately denser than the prototype.  The
    # positions form loose rings and preserve open centres for navigation.
    for index, name in enumerate(
        ("landing", "mosswater", "ember", "fernreach", "flooded", "cinder")
    ):
        tree_types = (0, 3, 4, 5) if name != "ember" and name != "cinder" else (1, 2, 5)
        for ring_index in range(14):
            angle = ring_index * 2.39996322973 + index * 0.41
            radius = 0.68 + (ring_index % 3) * 0.08
            x_factor = 0.5 + math.cos(angle) * radius * 0.42
            z_factor = 0.5 + math.sin(angle) * radius * 0.42
            add(
                *zone_point(name, x_factor, z_factor),
                ITEM_TREE,
                (tree_types[ring_index % len(tree_types)], 0, 0, 0),
            )

        for bush_index in range(10):
            angle = bush_index * 2.39996322973 + 0.7
            x_factor = 0.5 + math.cos(angle) * 0.30
            z_factor = 0.5 + math.sin(angle) * 0.30
            add(*zone_point(name, x_factor, z_factor), ITEM_BUSH)

        # An occasional boulder/mushroom breaks up the silhouette without
        # turning the traversal routes into an obstacle course.
        add(*zone_point(name, 0.24, 0.58), ITEM_BOULDER)
        add(*zone_point(name, 0.76, 0.42), ITEM_MUSHROOM)

    # Extra pickups are spaced along the branches: a modest early recovery
    # curve, then stronger weapons in the late lava and flooded sections.
    for name, powers in (
        ("mosswater", (POW_HEALTH, POW_LASER)),
        ("ember", (POW_HEALTH, POW_TRIBLAST)),
        ("fernreach", (POW_HEATSEEK, POW_SHIELD)),
        ("flooded", (POW_HEALTH, POW_NUKE)),
        ("cinder", (POW_SHIELD, POW_SONIC)),
    ):
        add(*zone_point(name, 0.34, 0.70), ITEM_POWERUP, (powers[0], 0, 0, 0))
        add(*zone_point(name, 0.68, 0.30), ITEM_POWERUP, (powers[1], 0, 0, 0))

    raw.sort(key=lambda item: (item[0], item[1]))
    return [pack_item(x, z, item_type, parm) for x, z, item_type, parm in raw]


def write_level(path: Path) -> None:
    assert MAP_W % SUPERTILE_SIZE == 0 and MAP_D % SUPERTILE_SIZE == 0
    original = (ORIGINAL_DIR / "Level1.ter").read_bytes()
    offsets = struct.unpack_from(">7i2h2i", original, 0)
    source_width, source_depth = offsets[7], offsets[8]
    tex_all = struct.unpack_from(f">{source_width * source_depth}H", original, offsets[0])
    hm_all = struct.unpack_from(f">{source_width * source_depth}H", original, offsets[1])
    original_path = struct.unpack_from(
        f">{source_width * source_depth}H", original, offsets[2]
    )
    hmtile_offset = offsets[5]
    tex_attributes = original[offsets[9] : offsets[10]]

    # Use true cliff texture values, not the random grass mix of the prototype.
    cliff_tex_samples = tuple(
        tex_all[index]
        for index, height_tile in enumerate(hm_all)
        if (height_tile & 0x0FFF) == HMTILE_CLIFF_TOP
        and original_path[index] == PATH_SOLID_ALL
    )
    assert cliff_tex_samples

    mask = make_walkable_mask()
    tex_layer: list[int] = []
    hm_layer: list[int] = []
    path_layer: list[int] = []
    for row in range(MAP_D):
        for col in range(MAP_W):
            if mask[row][col]:
                tex_layer.append(source_tex_value(tex_all, source_width, col, row))
                hm_layer.append(HMTILE_FLOOR)
                path_layer.append(0)
            else:
                tex_layer.append(cliff_tex_value(cliff_tex_samples, col, row))
                hm_layer.append(HMTILE_CLIFF_TOP)
                path_layer.append(PATH_SOLID_ALL)

    # Only two heightmap tiles are needed; output IDs are direct indices.
    hmtile_blob = b"".join(
        original[
            hmtile_offset + source_id * 1024 : hmtile_offset + (source_id + 1) * 1024
        ]
        for source_id in (HMTILE_FLOOR, HMTILE_CLIFF_TOP)
    )
    # Remap direct source IDs to this compact output list.
    hm_layer = [0 if value == HMTILE_FLOOR else 1 for value in hm_layer]

    items = build_items(mask)
    assert all(path_layer[col] == PATH_SOLID_ALL for col in range(MAP_W))
    assert all(
        path_layer[(MAP_D - 1) * MAP_W + col] == PATH_SOLID_ALL
        for col in range(MAP_W)
    )
    assert all(path_layer[row * MAP_W] == PATH_SOLID_ALL for row in range(MAP_D))
    assert all(
        path_layer[row * MAP_W + MAP_W - 1] == PATH_SOLID_ALL
        for row in range(MAP_D)
    )

    cell_count = MAP_W * MAP_D
    texture_offset = 40
    height_offset = texture_offset + cell_count * 2
    path_offset = height_offset + cell_count * 2
    item_offset = path_offset + cell_count * 2
    item_blob = struct.pack(">i", len(items)) + b"".join(items)
    hmtile_output_offset = item_offset + len(item_blob)
    tex_attribute_offset = hmtile_output_offset + len(hmtile_blob)
    tile_animation_offset = tex_attribute_offset + len(tex_attributes)

    header = struct.pack(
        ">7i2h2i",
        texture_offset,
        height_offset,
        path_offset,
        item_offset,
        0,
        hmtile_output_offset,
        0,
        MAP_W,
        MAP_D,
        tex_attribute_offset,
        tile_animation_offset,
    )
    output = b"".join(
        (
            header,
            struct.pack(f">{cell_count}H", *tex_layer),
            struct.pack(f">{cell_count}H", *hm_layer),
            struct.pack(f">{cell_count}H", *path_layer),
            item_blob,
            hmtile_blob,
            tex_attributes,
        )
    )
    path.write_bytes(output)
    print(
        f"wrote {path}: {MAP_W}x{MAP_D}, {len(items)} items, "
        f"{sum(sum(row) for row in mask)} walkable cells, contained edge"
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ORIGINAL_DIR / "Level1.trt", OUT_DIR / "Level1.trt")
    write_level(OUT_DIR / "Level1.ter")
    write_level(OUT_DIR / "Level1Pro.ter")


if __name__ == "__main__":
    main()
