from __future__ import annotations

import chess
import cv2
import numpy as np

from chess_plugin.db import Database
from chess_plugin.engine import ChessEngine
from chess_plugin.game import BAND_LEVEL_RANGES, ChessSession, Phase, SkillBand
from chess_plugin.ui import (
    build_band_scene,
    build_level_scene,
    build_play_overlay,
    build_profile_scene,
    build_promotion_scene,
    load_icon,
)
from chess_plugin.vision import BoardOccupancyDetector, hypothesize_moves, occupancy_from_board
from gamebot.adapters.camera import MockCamera
from gamebot.geometry import TableLayout
from gamebot.plugins.protocol import IdentifyContext, RuntimeHandles
from gamebot.ui import Scene


def looks_like_chessboard(frame_bgr: np.ndarray, corners: np.ndarray | None = None) -> bool:
    """Heuristic identify: 8x8 alternating pattern in the board ROI."""
    if corners is not None and len(corners) == 4:
        from chess_plugin.vision import warp_board

        warped = warp_board(frame_bgr, corners, out_size=240)
    else:
        h, w = frame_bgr.shape[:2]
        side = int(min(w, h) * 0.55)
        x0 = (w - side) // 2
        y0 = (h - side) // 2
        warped = frame_bgr[y0 : y0 + side, x0 : x0 + side]
        if warped.size == 0:
            return False
        warped = cv2.resize(warped, (240, 240))

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # OpenCV chessboard finder (inner corners 7x7)
    found, _ = cv2.findChessboardCorners(gray, (7, 7), flags=cv2.CALIB_CB_ADAPTIVE_THRESH)
    if found:
        return True
    # Fallback: checker contrast between neighbouring cells
    sq = 240 // 8
    means = []
    for r in range(8):
        row = []
        for c in range(8):
            patch = gray[r * sq : (r + 1) * sq, c * sq : (c + 1) * sq]
            row.append(float(patch.mean()) if patch.size else 0.0)
        means.append(row)
    flips = 0
    total = 0
    for r in range(8):
        for c in range(7):
            total += 1
            if abs(means[r][c] - means[r][c + 1]) > 15:
                flips += 1
    return total > 0 and flips / total > 0.55


class ChessGameSession:
    def __init__(self, runtime: RuntimeHandles) -> None:
        self.runtime = runtime
        self.layout: TableLayout = runtime.layout
        cfg = runtime.config
        db_path = runtime.data_dir / "chess.db"
        self.db = Database(db_path)
        self.session = ChessSession(self.db)
        self.engine = ChessEngine(
            getattr(cfg, "engine_path", None),
            getattr(cfg, "engine_threads", 2),
            getattr(cfg, "engine_hash_mb", 64),
        )
        self.occupancy = BoardOccupancyDetector(self.layout)
        self._prev_occ = None
        self._waiting_engine = False
        self._exit = False
        self._camera = getattr(runtime, "camera", None)
        self._calibration = getattr(runtime, "calibration", None)

    def on_enter(self) -> None:
        self.engine.open()
        self.session.phase = Phase.PROFILE

    def wants_exit(self) -> bool:
        return self._exit

    def build_scene(self) -> Scene:
        s = self.session
        layout = self.layout
        if s.phase == Phase.PROFILE:
            profiles = self.db.list_profiles()
            names = [(f"profile:{p.id}", p.name) for p in profiles]
            names.append(("profile:new", "New profile"))
            return build_profile_scene(layout, names)
        if s.phase == Phase.SKILL_BAND:
            return build_band_scene(layout)
        if s.phase == Phase.SKILL_LEVEL:
            return build_level_scene(layout, list(BAND_LEVEL_RANGES[s.skill_band]))
        if s.phase == Phase.PROMOTE:
            return build_promotion_scene(layout)
        if s.phase == Phase.GAME_OVER:
            sc = Scene(layout)
            sc.title = f"Game over: {s.board.result()}"
            sc.add_button("again", "Again", layout.margin_mm + layout.board_size_mm + 10, layout.margin_mm, 100, 40)
            sc.add_button("exit", "Exit", layout.margin_mm + layout.board_size_mm + 10, layout.margin_mm + 60, 100, 40)
            return sc
        checks = [(h.from_sq, h.to_sq) for h in s.check_highlights()]
        revert = None
        if s.revert_hint is not None:
            revert = (s.revert_hint.from_sq, s.revert_hint.to_sq)
        promo_label = None
        if s.last_engine_promotion:
            promo_label = chess.piece_symbol(s.last_engine_promotion).upper()
        return build_play_overlay(layout, s.last_engine_move, checks, revert, promo_label)

    def handle_zone(self, zone_id: str) -> None:
        s = self.session
        if zone_id.startswith("profile:"):
            rest = zone_id.split(":", 1)[1]
            if rest == "new":
                p = self.db.create_profile(f"Player{len(self.db.list_profiles())}", "intermediate", 5)
            else:
                p = self.db.get_profile(int(rest))
            if p:
                s.select_profile(p)
            return
        if zone_id.startswith("band:"):
            s.select_skill_band(SkillBand(zone_id.split(":", 1)[1]))
            return
        if zone_id.startswith("level:"):
            s.select_skill_level(int(zone_id.split(":", 1)[1]))
            self._prev_occ = occupancy_from_board(s.board)
            return
        if zone_id.startswith("promo:"):
            s.confirm_promotion(int(zone_id.split(":", 1)[1]))
            self._prev_occ = occupancy_from_board(s.board)
            return
        if zone_id == "again":
            s.phase = Phase.PROFILE
            return
        if zone_id == "exit":
            self._exit = True
            self.engine.close()
            self.db.close()

    def handle_key(self, key: int) -> None:
        if key == ord("m"):
            self._simulate_human_move()
        elif key == ord("e"):
            self._do_engine_turn()

    def tick(self, frame_bgr: np.ndarray) -> None:
        s = self.session
        if s.phase == Phase.PLAY and s.is_human_turn():
            self._poll_board_move(frame_bgr)
        if s.phase == Phase.PLAY and not s.is_human_turn() and not self._waiting_engine:
            self._waiting_engine = True
            self._do_engine_turn()
            self._waiting_engine = False

    def _do_engine_turn(self) -> None:
        s = self.session
        if s.phase != Phase.PLAY or s.is_human_turn():
            return
        thinking = getattr(self.runtime, "thinking", None)
        if callable(thinking):
            thinking(True)
        try:
            result = self.engine.play(s.board, s.skill_level)
            s.apply_engine_move(result.move)
            cam = self._camera
            if isinstance(cam, MockCamera):
                cam.set_occupancy(occupancy_from_board(s.board))
            self._prev_occ = occupancy_from_board(s.board)
        finally:
            if callable(thinking):
                thinking(False)

    def _simulate_human_move(self) -> None:
        s = self.session
        if not s.is_human_turn():
            return
        moves = list(s.board.legal_moves)
        if not moves:
            return
        move = next((m for m in moves if not s.board.is_capture(m)), moves[0])
        s.try_human_move(move)
        cam = self._camera
        if isinstance(cam, MockCamera):
            cam.set_occupancy(occupancy_from_board(s.board))
        self._prev_occ = occupancy_from_board(s.board)

    def _poll_board_move(self, frame_bgr: np.ndarray) -> None:
        s = self.session
        cal = self._calibration
        if not s.is_human_turn() or cal is None:
            return
        result = self.occupancy.detect(frame_bgr, cal.board_corners_camera)
        if self._prev_occ is None:
            self._prev_occ = result.grid
            return
        hyps = hypothesize_moves(self._prev_occ, result.grid, s.board)
        if not hyps:
            return
        if len(hyps) > 1 and hyps[0].score - hyps[1].score < 0.05:
            return
        if s.try_human_move(hyps[0].move):
            self._prev_occ = occupancy_from_board(s.board)


class ChessPlugin:
    id = "chess"
    name = "Chess"

    def icon(self) -> np.ndarray:
        return load_icon()

    def identify(self, frame_bgr: np.ndarray, ctx: IdentifyContext) -> bool:
        return looks_like_chessboard(frame_bgr, ctx.board_corners_camera)

    def create_session(self, runtime: RuntimeHandles) -> ChessGameSession:
        return ChessGameSession(runtime)


def get_plugin() -> ChessPlugin:
    return ChessPlugin()
