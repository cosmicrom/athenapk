#!/usr/bin/env python3
"""Plot mesh blocks from an Athena++ mesh_structure.dat file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


def read_blocks(path: Path) -> list[list[tuple[float, float]]]:
    blocks: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    with path.open() as mesh_file:
        for line in mesh_file:
            line = line.strip()
            if line.startswith("#MeshBlock"):
                if current:
                    blocks.append(current)
                    current = []
            elif line and not line.startswith("#"):
                x, y, *_ = map(float, line.split())
                current.append((x, y))

    if current:
        blocks.append(current)
    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot an Athena++ mesh structure")
    parser.add_argument("file", nargs="?", default="mesh_structure.dat")
    parser.add_argument("-o", "--output", help="Save the plot instead of displaying it")
    args = parser.parse_args()

    path = Path(args.file)
    blocks = read_blocks(path)
    if not blocks:
        raise SystemExit(f"No mesh coordinates found in {path}")

    _, ax = plt.subplots()
    for block in blocks:
        x, y = zip(*block)
        ax.plot(x, y, "k-")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(path.name)
    ax.autoscale()

    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
