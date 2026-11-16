"""Minimal pure-Python GROMACS .xtc reader (no mdtraj/MDAnalysis in the env).

A direct port of xdrfile.c's xdrfile_decompress_coord_float (magic 1995 frames).
Frames are length-prefixed, so frames we do not want are SKIPPED without being
decoded — sampling every k-th frame costs ~1/k of a full decode.

Validated against ATLAS's own per-residue RMSF tables (see md_ensemble.validate).

    for i, xyz_angstrom in read_xtc(path, every=10): ...
"""
from __future__ import annotations

import struct

import numpy as np

MAGIC = 1995
FIRSTIDX = 9
MAGICINTS = [
    0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 10, 12, 16, 20, 25, 32, 40, 50, 64,
    80, 101, 128, 161, 203, 256, 322, 406, 512, 645, 812, 1024, 1290,
    1625, 2048, 2580, 3250, 4096, 5060, 6501, 8192, 10321, 13003,
    16384, 20642, 26007, 32768, 41285, 52015, 65536, 82570, 104031,
    131072, 165140, 208063, 262144, 330280, 416127, 524287, 660561,
    832255, 1048576, 1321122, 1664510, 2097152, 2642245, 3329021,
    4194304, 5284491, 6658042, 8388607, 10568983, 13316085, 16777216]


class _Bits:
    """MSB-first bit reader over a bytes object (== xdrfile receivebits)."""
    __slots__ = ("b", "pos")

    def __init__(self, b: bytes):
        self.b, self.pos = b, 0

    def get(self, n: int) -> int:
        if n == 0:
            return 0
        p = self.pos
        i0, i1 = p >> 3, (p + n + 7) >> 3
        chunk = int.from_bytes(self.b[i0:i1], "big")
        shift = (i1 << 3) - (p + n)
        self.pos = p + n
        return (chunk >> shift) & ((1 << n) - 1)


def _receiveints(bits: _Bits, nbits: int, sizes) -> list[int]:
    # bytes[0] is the LEAST significant byte of one big integer
    bs, k = [], nbits
    while k > 8:
        bs.append(bits.get(8)); k -= 8
    if k > 0:
        bs.append(bits.get(k))
    big = 0
    for j in range(len(bs) - 1, -1, -1):
        big = (big << 8) | bs[j]
    z = big % sizes[2]; big //= sizes[2]
    y = big % sizes[1]; big //= sizes[1]
    return [big, y, z]


def _sizeofint(size: int) -> int:
    n, num = 0, 1
    while size >= num and n < 32:
        n += 1; num <<= 1
    return n


def _decode(buf: bytes, natoms, precision, minint, maxint, smallidx) -> np.ndarray:
    sizeint = [maxint[k] - minint[k] + 1 for k in range(3)]
    if any(s > 0xFFFFFF for s in sizeint):
        bitsizeint = [_sizeofint(s) for s in sizeint]
        bitsize = 0
    else:
        bitsize = (sizeint[0] * sizeint[1] * sizeint[2]).bit_length()
    smaller = MAGICINTS[max(FIRSTIDX, smallidx - 1)] // 2
    smallnum = MAGICINTS[smallidx] // 2
    sizesmall = [MAGICINTS[smallidx]] * 3
    bits = _Bits(buf)
    out = np.empty((natoms, 3), np.float64)
    o = 0
    i = run = 0
    while i < natoms:
        if bitsize == 0:
            tc = [bits.get(bitsizeint[0]), bits.get(bitsizeint[1]), bits.get(bitsizeint[2])]
        else:
            tc = _receiveints(bits, bitsize, sizeint)
        i += 1
        tc = [tc[0] + minint[0], tc[1] + minint[1], tc[2] + minint[2]]
        prev = tc[:]
        flag = bits.get(1)
        is_smaller = 0
        if flag == 1:
            run = bits.get(5)
            is_smaller = run % 3
            run -= is_smaller
            is_smaller -= 1
        if run > 0:
            for k in range(0, run, 3):
                t = _receiveints(bits, smallidx, sizesmall)
                i += 1
                t = [t[0] + prev[0] - smallnum, t[1] + prev[1] - smallnum,
                     t[2] + prev[2] - smallnum]
                if k == 0:                      # first/second atom swapped (water trick)
                    t, prev = prev, t
                    out[o] = prev; o += 1
                else:
                    prev = t[:]
                out[o] = t; o += 1
        else:
            out[o] = tc; o += 1
        smallidx += is_smaller
        if is_smaller < 0:
            smallnum = smaller
            smaller = MAGICINTS[smallidx - 1] // 2 if smallidx > FIRSTIDX else 0
        elif is_smaller > 0:
            smaller = smallnum
            smallnum = MAGICINTS[smallidx] // 2
        sizesmall = [MAGICINTS[smallidx]] * 3
    if o != natoms:
        raise ValueError(f"xtc decode produced {o} atoms, expected {natoms}")
    return out / precision


def read_xtc(path, every: int = 1, start: int = 0, max_frames: int | None = None):
    """Yield (frame_index, coords[natoms,3] in Å) for frames start, start+every, …"""
    with open(path, "rb") as fh:
        data = fh.read()
    p, idx, n_out = 0, 0, 0
    L = len(data)
    while p + 56 <= L:
        magic, natoms, _step = struct.unpack_from(">iii", data, p)
        if magic != MAGIC:
            raise ValueError(f"unsupported xtc magic {magic} at byte {p}")
        p += 16 + 36                                   # header (+time) + box
        (lsize,) = struct.unpack_from(">i", data, p); p += 4
        want = idx >= start and (idx - start) % every == 0
        if lsize <= 9:
            xyz = np.array(struct.unpack_from(f">{3 * lsize}f", data, p)).reshape(-1, 3)
            p += 12 * lsize
        else:
            precision, = struct.unpack_from(">f", data, p)
            mi = struct.unpack_from(">3i", data, p + 4)
            ma = struct.unpack_from(">3i", data, p + 16)
            smallidx, nbytes = struct.unpack_from(">2i", data, p + 28)
            p += 36
            xyz = (_decode(data[p:p + nbytes], lsize, precision, mi, ma, smallidx)
                   if want else None)
            p += (nbytes + 3) & ~3                     # XDR opaque: pad to 4
        if want:
            yield idx, xyz * 10.0                      # nm -> Å
            n_out += 1
            if max_frames and n_out >= max_frames:
                return
        idx += 1
