#!/usr/bin/env python3
"""Compare two PNG images pixel by pixel, or print the pixel hash of one (Python 3.10+, standard library only).

Why this exists: the PNG files written by the engine carry the generation parameters as text, so two files with
identical pixels can still have different file hashes. Every "same image" and every PSNR number in
docs/MEASUREMENTS.md was produced by comparing decoded pixels, and this script lets you repeat that.

Usage:
  python scripts/compare.py IMAGE.png              -> PIXEL_SHA256 <hex> <width>x<height>
  python scripts/compare.py A.png B.png            -> SAME_PIXELS, or DIFFERENT with max difference and PSNR

Output of a comparison:
  file_sha256_equal  true|false     (whole file, text chunks included)
  pixel_sha256_a / pixel_sha256_b   (sha256 of the decoded RGB bytes, row by row)
  max_abs_diff       0..255         (largest difference of any channel of any pixel)
  mean_abs_diff      average difference per channel
  psnr_db            10*log10(255^2 / mean squared error); "inf" when the pixels are identical
  last line          SAME_PIXELS or DIFFERENT
Exit code 0 when the pixels are identical (or for a single image), 1 when they differ, 2 on a usage or format error.

Supported: 8-bit RGB or RGBA PNG, not interlaced (what sd-cli writes). Alpha is ignored.
"""
import hashlib
import math
import struct
import sys
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PngError(ValueError):
    pass


def read_png_rgb(path):
    """Decode an 8-bit RGB/RGBA non-interlaced PNG. Returns (width, height, bytes of RGB rows)."""
    data = Path(path).read_bytes()
    if data[:8] != PNG_SIGNATURE:
        raise PngError("%s is not a PNG file" % path)
    pos = 8
    header = None
    idat = []
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if header is None:
        raise PngError("%s has no IHDR chunk" % path)
    width, height, depth, color, _compression, _filter, interlace = header
    if depth != 8 or color not in (2, 6) or interlace != 0:
        raise PngError("%s: only 8-bit RGB/RGBA, non-interlaced PNG is supported "
                       "(bit depth %d, color type %d, interlace %d)" % (path, depth, color, interlace))
    bpp = 3 if color == 2 else 4
    stride = width * bpp
    raw = zlib.decompress(b"".join(idat))
    if len(raw) != height * (stride + 1):
        raise PngError("%s: unexpected amount of image data" % path)
    out = bytearray(height * width * 3)
    prev = bytearray(stride)
    for y in range(height):
        start = y * (stride + 1)
        ftype = raw[start]
        line = bytearray(raw[start + 1:start + 1 + stride])
        if ftype == 1:                                   # Sub
            for i in range(bpp, stride):
                line[i] = (line[i] + line[i - bpp]) & 255
        elif ftype == 2:                                 # Up
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif ftype == 3:                                 # Average
            for i in range(stride):
                left = line[i - bpp] if i >= bpp else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif ftype == 4:                                 # Paeth
            for i in range(stride):
                a = line[i - bpp] if i >= bpp else 0
                b = prev[i]
                c = prev[i - bpp] if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pred) & 255
        elif ftype != 0:
            raise PngError("%s: unknown PNG filter %d" % (path, ftype))
        row = y * width * 3
        if bpp == 3:
            out[row:row + stride] = line
        else:
            out[row:row + width * 3:3] = line[0::4]
            out[row + 1:row + width * 3:3] = line[1::4]
            out[row + 2:row + width * 3:3] = line[2::4]
        prev = line
    return width, height, bytes(out)


def pixel_sha256(path):
    """sha256 of the decoded RGB bytes: the same for two files with the same pixels, whatever their text chunks."""
    width, height, rgb = read_png_rgb(path)
    return hashlib.sha256(rgb).hexdigest(), width, height


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(path_a, path_b):
    wa, ha, a = read_png_rgb(path_a)
    wb, hb, b = read_png_rgb(path_b)
    if (wa, ha) != (wb, hb):
        raise PngError("sizes differ: %dx%d and %dx%d" % (wa, ha, wb, hb))
    result = {
        "width": wa, "height": ha,
        "file_sha256_equal": file_sha256(path_a) == file_sha256(path_b),
        "pixel_sha256_a": hashlib.sha256(a).hexdigest(),
        "pixel_sha256_b": hashlib.sha256(b).hexdigest(),
    }
    if a == b:
        result.update({"max_abs_diff": 0, "mean_abs_diff": 0.0, "psnr_db": math.inf, "same_pixels": True})
        return result
    counts = [0] * 256                                   # histogram of absolute differences
    for x, y in zip(a, b):
        counts[x - y if x >= y else y - x] += 1
    total = len(a)
    max_diff = max(d for d in range(256) if counts[d])
    mean = sum(d * n for d, n in enumerate(counts)) / total
    mse = sum(d * d * n for d, n in enumerate(counts)) / total
    result.update({"max_abs_diff": max_diff, "mean_abs_diff": round(mean, 4),
                   "psnr_db": round(10 * math.log10(255 * 255 / mse), 2), "same_pixels": False})
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) not in (1, 2) or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    try:
        if len(argv) == 1:
            digest, width, height = pixel_sha256(argv[0])
            print("PIXEL_SHA256 %s %dx%d" % (digest, width, height))
            return 0
        r = compare(argv[0], argv[1])
    except (OSError, PngError, zlib.error) as e:
        print("ERROR: %s. Fix: pass PNG files written by scripts/generate.py (see docs/MEASUREMENTS.md)." % e)
        return 2
    print("file_sha256_equal %s" % str(r["file_sha256_equal"]).lower())
    print("pixel_sha256_a    %s" % r["pixel_sha256_a"])
    print("pixel_sha256_b    %s" % r["pixel_sha256_b"])
    print("max_abs_diff      %d" % r["max_abs_diff"])
    print("mean_abs_diff     %s" % r["mean_abs_diff"])
    print("psnr_db           %s" % ("inf" if r["same_pixels"] else r["psnr_db"]))
    print("SAME_PIXELS" if r["same_pixels"] else "DIFFERENT")
    return 0 if r["same_pixels"] else 1


if __name__ == "__main__":
    sys.exit(main())
