"""Serialization writers for .ter files, Map.tga, and level_preview.png."""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image, ImageDraw

from .constants import (
    D,
    IT_CRYSTAL,
    IT_EGG,
    IT_GASVENT,
    IT_LAVA,
    IT_PORTAL,
    IT_POWERUP,
    IT_PTERA,
    IT_REX,
    IT_ROLLBOULDER,
    IT_SPITTER,
    IT_START,
    IT_STEGO,
    IT_STEPSTONE,
    IT_TRICER,
    IT_WATER,
    TILE_PX,
    W,
)


def pack_items(items):
    """Pack terrain items into big-endian binary format matching Terrain.h."""
    blob = struct.pack(">i", len(items))
    for x, z, t, p, _ in items:
        blob += struct.pack(">hhh4bHii", x, z, t, p[0], p[1], p[2], p[3], 0, 0, 0)
    return blob


def write_ter(path, tex, hm_grid, path_layer, items, tile_bank, orig):
    """Write a complete .ter terrain file and return the deduplicated hm tile count."""
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
    header = struct.pack(
        ">7i2h2i", tex_off, hm_off, path_off, obj_off, 0,
        hmtile_off, 0, W, D, attrib_off, anim_off,
    )
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
    """Write RLE colormapped 8-bit GPS map matching TGA.c expectations."""
    rgb = orig.tile_avg[np.asarray(tex) & 0x0FFF]
    img = Image.fromarray(rgb, "RGB").convert("P", palette=Image.ADAPTIVE, colors=255)
    idx = np.asarray(img, np.uint8)
    ncol = int(idx.max()) + 1
    pal = img.getpalette()[:ncol * 3]
    header = struct.pack(
        "<BBBHHBHHHHBB",
        0, 1, 9, 0, ncol, 24, 0, D, W, D, 8, 0x20,
    )
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
    """Render annotated 4x preview PNG for visual inspection."""
    S = 4
    rgb = orig.tile_avg[np.asarray(tex) & 0x0FFF].astype(np.float32)
    shade = 0.55 + 0.45 * (1.0 - np.abs(mean_map - 85.0) / 170.0)
    shade = np.clip(shade, 0.4, 1.0)
    rgb = (rgb * shade[..., None]).astype(np.uint8)
    big = np.kron(rgb, np.ones((S, S, 1), np.uint8))
    img = Image.fromarray(big, "RGB")
    dr = ImageDraw.Draw(img)
    egg_colors = ["#00ff40", "#ffa000", "#ff2020", "#00e0ff", "#ff00ff"]
    enemy_colors = {
        IT_REX: "#c00000",
        IT_TRICER: "#a05820",
        IT_PTERA: "#c060ff",
        IT_STEGO: "#90a800",
        IT_SPITTER: "#3050ff",
    }
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
