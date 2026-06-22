from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def export_figure_png(figure: Figure, path: str | Path, dpi: int = 220) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=dpi, bbox_inches="tight")

