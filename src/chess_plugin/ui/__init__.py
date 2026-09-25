from __future__ import annotations

from pathlib import Path

import chess
import cv2
import numpy as np

from gamebot.geometry import TableLayout
from gamebot.ui import Scene


def build_profile_scene(layout: TableLayout, names: list[tuple[str, str]]) -> Scene:
    scene = Scene(layout)
    scene.title = "Choose profile"
    ox = layout.margin_mm + layout.board_size_mm + 10
    y = layout.margin_mm
    for i, (zid, label) in enumerate(names):
        scene.add_button(zid, label, ox, y + i * 50, layout.menu_width_mm - 10, 40)
    return scene


def build_band_scene(layout: TableLayout) -> Scene:
    scene = Scene(layout)
    scene.title = "Skill range"
    ox = layout.margin_mm + layout.board_size_mm + 10
    y = layout.margin_mm
    for i, band in enumerate(["beginner", "intermediate", "advanced", "expert"]):
        scene.add_button(f"band:{band}", band.title(), ox, y + i * 50, layout.menu_width_mm - 10, 40)
    return scene


def build_level_scene(layout: TableLayout, levels: list[int]) -> Scene:
    scene = Scene(layout)
    scene.title = "Skill level"
    ox = layout.margin_mm + layout.board_size_mm + 10
    y = layout.margin_mm
    for i, lvl in enumerate(levels):
        scene.add_button(f"level:{lvl}", f"Level {lvl}", ox, y + i * 50, layout.menu_width_mm - 10, 40)
    return scene


PROMO_LABELS = {
    chess.QUEEN: "Q",
    chess.ROOK: "R",
    chess.BISHOP: "B",
    chess.KNIGHT: "N",
}


def build_promotion_scene(layout: TableLayout) -> Scene:
    scene = Scene(layout)
    scene.title = "Promote"
    ox = layout.margin_mm + layout.board_size_mm + 10
    y = layout.margin_mm
    for i, (pt, lab) in enumerate(PROMO_LABELS.items()):
        scene.add_button(f"promo:{pt}", lab, ox, y + i * 55, layout.menu_width_mm - 10, 45, piece=pt)
    return scene


def build_play_overlay(
    layout: TableLayout,
    engine_move: chess.Move | None,
    check_pairs: list[tuple[chess.Square, chess.Square]],
    revert: tuple[chess.Square, chess.Square] | None,
    engine_promo_label: str | None,
) -> Scene:
    scene = Scene(layout)
    scene.title = "Play"
    if engine_move is not None:
        x0, y0 = layout.square_center(engine_move.from_square)
        x1, y1 = layout.square_center(engine_move.to_square)
        scene.add_arrow(x0, y0, x1, y1, (0, 220, 255))
    for a, b in check_pairs:
        x0, y0 = layout.square_center(a)
        x1, y1 = layout.square_center(b)
        scene.add_arrow(x0, y0, x1, y1, (0, 0, 255))
    if revert is not None:
        x0, y0 = layout.square_center(revert[0])
        x1, y1 = layout.square_center(revert[1])
        scene.add_arrow(x0, y0, x1, y1, (0, 0, 255))
        scene.add_text("Illegal — move back", layout.margin_mm, layout.margin_mm - 30, (0, 0, 255))
    if engine_promo_label:
        ox = layout.margin_mm + layout.board_size_mm + 40
        oy = layout.margin_mm + 40
        scene.add_icon(engine_promo_label, ox, oy)
        scene.add_text(f"Bot promoted to {engine_promo_label}", ox - 20, oy + 50)
    return scene


def load_icon() -> np.ndarray:
    path = Path(__file__).resolve().parent.parent / "assets" / "icon.png"
    img = cv2.imread(str(path))
    if img is not None:
        return img
    return _fallback_icon()


def _fallback_icon() -> np.ndarray:
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[:] = (40, 90, 40)
    for r in range(8):
        for c in range(8):
            if (r + c) % 2:
                cv2.rectangle(img, (c * 16, r * 16), ((c + 1) * 16, (r + 1) * 16), (220, 220, 220), -1)
            else:
                cv2.rectangle(img, (c * 16, r * 16), ((c + 1) * 16, (r + 1) * 16), (40, 70, 40), -1)
    cv2.putText(img, "C", (44, 84), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 3)
    return img
