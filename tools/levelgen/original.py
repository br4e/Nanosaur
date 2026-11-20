"""Reference data loader for original Level1.ter and Level1.trt."""

from __future__ import annotations

import struct

import numpy as np

from .constants import D, ORIG_DIR, W


class Original:
    """Loaded original terrain tiles, attributes, heightmaps, and average RGBs."""

    def __init__(self):
        data = (ORIG_DIR / "Level1.ter").read_bytes()
        h = struct.unpack_from(">7i2h2i", data, 0)
        (
            tex_off, hm_off, path_off, obj_off, _, hmtile_off, _,
            ow, od, attrib_off, anim_off,
        ) = h
        assert (ow, od) == (W, D), "original size mismatch"
        n = ow * od
        self.tex = np.frombuffer(data, ">u2", n, tex_off).reshape(od, ow)
        self.hm = np.frombuffer(data, ">u2", n, hm_off).reshape(od, ow)
        self.path = np.frombuffer(data, ">u2", n, path_off).reshape(od, ow)
        n_hm_tiles = (attrib_off - hmtile_off) // 1024
        self.hm_tiles = np.frombuffer(
            data, "u1", n_hm_tiles * 1024, hmtile_off
        ).reshape(n_hm_tiles, 32, 32)
        self.attrib_blob = data[attrib_off:anim_off]
        self.anim_blob = data[anim_off:]
        self.attribs = [
            struct.unpack_from(">Hhbbh", self.attrib_blob, i * 8)
            for i in range(len(self.attrib_blob) // 8)
        ]

        trt = (ORIG_DIR / "Level1.trt").read_bytes()
        n_tex = struct.unpack_from(">i", trt, 0)[0]
        texels = np.frombuffer(trt, ">u2", n_tex * 32 * 32, 4).reshape(n_tex, 32, 32)
        r = ((texels >> 10) & 31) * 255 // 31
        g = ((texels >> 5) & 31) * 255 // 31
        b = (texels & 31) * 255 // 31
        self.tile_avg = np.stack(
            [r.mean(axis=(1, 2)), g.mean(axis=(1, 2)), b.mean(axis=(1, 2))],
            axis=1,
        ).astype(np.uint8)
        self.n_tex_tiles = n_tex
