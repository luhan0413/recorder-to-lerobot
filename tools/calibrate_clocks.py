"""Fit affine maps from independently matched native/common clock events."""

import argparse
import json
from pathlib import Path

import numpy as np


def fit_clock_maps(anchors, max_residual=0.005):
    result = {}
    for episode, clocks in anchors.items():
        result[episode] = {}
        for column, spec in clocks.items():
            native = np.asarray(spec["native"], dtype=float)
            common = np.asarray(spec["common_host_monotonic"], dtype=float)
            if native.ndim != 1 or native.shape != common.shape or len(native) < 3:
                raise ValueError(f"{episode}/{column}: need >=3 matched anchor pairs")
            if (
                not np.isfinite(native).all()
                or not np.isfinite(common).all()
                or (np.diff(native) <= 0).any()
                or (np.diff(common) <= 0).any()
            ):
                raise ValueError(f"{column}: invalid anchors or clock reset")
            if not spec.get("provenance"):
                raise ValueError("independent anchor provenance required")
            x = native - native[0]
            y = common - common[0]
            design = np.column_stack([x, np.ones(len(x))])
            scale, centered_offset = np.linalg.lstsq(design, y, rcond=None)[0]
            offset = common[0] + centered_offset - scale * native[0]
            residual = float(np.max(abs(native * scale + offset - common)))
            if scale <= 0 or residual > max_residual:
                raise ValueError(
                    f"{column}: invalid scale or residual {residual}s exceeds {max_residual}s"
                )
            result[episode][column] = {
                "scale": float(scale),
                "offset": float(offset),
                "provenance": spec["provenance"],
                "max_anchor_residual_seconds": residual,
                "anchor_native_range": [float(native[0]), float(native[-1])],
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-residual", type=float, default=0.005)
    args = parser.parse_args()
    try:
        if not np.isfinite(args.max_residual) or args.max_residual < 0:
            raise ValueError("invalid max residual")
        maps = fit_clock_maps(json.loads(args.anchors.read_text()), args.max_residual)
        with args.output.open("x") as file:
            json.dump(maps, file, indent=2)
            file.write("\n")
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
