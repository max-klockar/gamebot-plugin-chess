from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import chess
import cv2
import numpy as np

from chess_plugin.db import Database
from chess_plugin.engine import ChessEngine
from chess_plugin.game import BAND_LEVEL_RANGES, ChessSession, Phase, SkillBand, elo_to_skill
from chess_plugin.ui import (
    build_band_scene,
    build_fault_scene,
    build_level_scene,
    build_play_overlay,
    build_profile_scene,
    build_promotion_scene,
    load_icon,
)
from gamebot.adapters.led import LedMode
from chess_plugin.vision import (
    BoardOccupancyDetector,
    OccupancyKind,
    classify_occupancy_vs_board,
    detect_human_is_white,
    detect_human_is_white_from_colors,
    looks_like_starting_occupancy,
    occupancy_debug_summary,
    occupancy_frame_is_noise,
    occupancy_from_board,
    occupancy_mismatch,
    setup_from_corner_pieces,
    square_rc,
    warp_board,
)
from gamebot.adapters.camera import MockCamera
from gamebot.calibration import estimate_homography
from gamebot.geometry import TableLayout, force_square_quad, warp_points
from gamebot.plugins.protocol import IdentifyContext, RuntimeHandles
from gamebot.ui import Scene


def looks_like_chessboard(frame_bgr: np.ndarray, corners: np.ndarray | None = None) -> bool:
    """Heuristic identify: 8x8 alternating pattern in the board ROI."""
    if corners is not None and len(corners) == 4:
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
        self._last_gray: np.ndarray | None = None
        self._waiting_engine = False
        self._awaiting_settle = False
        self._pending_uci: str | None = None
        self._pending_count = 0
        self._pending_fault_key: str | None = None
        self._pending_fault_count = 0
        self._fault_clear_count = 0
        self._dbg_tick = 0
        self._exit = False
        self._confirm_need = 2
        self._fault_confirm_need = 10
        self._fault_clear_need = 4
        self._state_log_path = Path(runtime.data_dir) / "chess_state.log"
        self._state_json_path = Path(runtime.data_dir) / "chess_state.json"
        self._camera = getattr(runtime, "camera", None)
        self._calibration = getattr(runtime, "calibration", None)
        self._setup_ready_frames = 0
        self._setup_lock_every = 10
        self._setup_tick = 0
        self._lock_border_proj_px: np.ndarray | None = None
        self._last_good_quad: np.ndarray | None = None
        self._last_good_method: str | None = None
        self._lock_misses = 0
        self._lock_pending_quad: np.ndarray | None = None
        self._lock_pending_count = 0
        self._lock_frozen = False
        self._arrow_hidden = False
        self._capture_moves = bool(getattr(cfg, "capture_moves", False))
        self._pending_still: tuple[str, str] | None = None

    def _set_led(self, mode: LedMode) -> None:
        led = getattr(self.runtime, "led", None)
        if led is not None and hasattr(led, "set_mode"):
            led.set_mode(mode)

    def _log_state(self, event: str, **extra: object) -> None:
        """Append a debug line + refresh chess_state.json (FEN / turn / settle)."""
        s = self.session
        turn = "white" if s.board.turn == chess.WHITE else "black"
        whose = "human" if s.is_human_turn() else ("bot" if s.phase == Phase.PLAY else s.phase.name)
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            "phase": s.phase.name,
            "fen": s.board.fen(),
            "turn": turn,
            "to_move": whose,
            "human_is_white": s.human_is_white,
            "last_engine": s.last_engine_move.uci() if s.last_engine_move else None,
            "awaiting_settle": self._awaiting_settle,
            "pending_uci": self._pending_uci,
            "pending_count": self._pending_count,
            "pending_fault": self._pending_fault_key,
            "revert": (
                f"{chess.square_name(s.revert_hint.from_sq)}"
                f"{chess.square_name(s.revert_hint.to_sq)}:{s.revert_hint.reason}"
                if s.revert_hint
                else None
            ),
            "moves": [m.uci() for m in s.board.move_stack],
        }
        payload.update(extra)
        line = json.dumps(payload, ensure_ascii=True)
        try:
            self._state_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._state_log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._state_json_path.write_text(line + "\n", encoding="utf-8")
        except OSError as exc:
            if self._dbg_tick % 100 == 0:
                print(f"state log write failed: {exc}")
        print(
            f"state: {event} | {s.phase.name} {whose} FEN={s.board.board_fen()} "
            f"settle={self._awaiting_settle} eng={payload['last_engine']}"
        )

    def _save_move_still(self, frame_bgr: np.ndarray, *, who: str, uci: str) -> None:
        """Keep the confirming camera frame. Training pull already copies 04_play."""
        if not self._capture_moves:
            return
        from gamebot.capture_frames import save_move_still

        roi = self._board_roi()
        corners = None if roi is None else np.asarray(roi, dtype=float).reshape(-1, 2).tolist()
        zoom = float(getattr(self.runtime.config, "camera_zoom", 1.2))
        out = Path(self.runtime.data_dir) / "captures" / "sets" / "04_play"
        try:
            path = save_move_still(
                frame_bgr,
                out,
                uci=uci,
                who=who,
                fen=self.session.board.fen(),
                camera_zoom=zoom,
                board_corners_camera=corners,
            )
        except (OSError, RuntimeError) as exc:
            print(f"move capture failed: {exc}")
            return
        print(f"captured {who} {uci} → {path}")

    def _enter_fault(self, reason: str, squares: list[chess.Square] | None = None) -> None:
        print(f"FAULT — {reason} (waiting until board matches again)")
        self._set_led(LedMode.ERROR)
        self._awaiting_settle = False
        self._reset_pending()
        if self.session.phase != Phase.FAULT:
            if squares and len(squares) >= 2:
                self.session.enter_fault(squares[0], squares[1], reason)
            elif squares and len(squares) == 1:
                self.session.enter_fault(squares[0], squares[0], reason)
            else:
                self.session.enter_fault(chess.A1, chess.H8, reason)
        self._log_state(
            "fault",
            reason=reason,
            changed=[chess.square_name(sq) for sq in (squares or [])],
            mismatch=None,
        )

    def _clear_fault_state(self) -> None:
        self.session.clear_fault()
        self._pending_fault_key = None
        self._pending_fault_count = 0
        self._fault_clear_count = 0
        self._pending_uci = None
        self._pending_count = 0
        self._sync_turn_led()
        print("fault cleared — board looks right again")
        self._log_state("fault_cleared")

    def _reset_pending(self) -> None:
        self._pending_uci = None
        self._pending_count = 0
        self._pending_fault_key = None
        self._pending_fault_count = 0
        self._fault_clear_count = 0

    def _board_roi(self) -> np.ndarray | None:
        cal = self._calibration
        if cal is None:
            return None
        if self._last_good_quad is not None:
            return self._last_good_quad
        return cal.board_corners_camera

    def _read_occupancy(self, frame_bgr: np.ndarray):
        roi = self._board_roi()
        if roi is None:
            return None
        return self.occupancy.detect(frame_bgr, roi, prefer_yolo=True)

    def _sync_turn_led(self) -> None:
        s = self.session
        if s.phase == Phase.FAULT:
            self._set_led(LedMode.ERROR)
        elif s.phase != Phase.PLAY:
            self._set_led(LedMode.IDLE)
        elif self._pending_uci and not self._pending_uci.startswith("__"):
            # First sight of a human move — bot colour immediately, before confirm.
            self._set_led(LedMode.BOT_TURN)
        elif self._awaiting_settle or s.is_human_turn():
            # Arrow is up, or it is the human's turn to move.
            self._set_led(LedMode.PLAYER_TURN)
        else:
            self._set_led(LedMode.BOT_TURN)

    def _seed_lock_from_calibration(self) -> None:
        """Paint magenta border from cal once — board does not move; stop chasing vision."""
        cal = self._calibration
        if cal is None or cal.board_corners_camera is None:
            return
        self._apply_board_corners(
            cal.board_corners_camera, force_square=False, update_projector=True, persist=False
        )
        self._last_good_quad = np.asarray(cal.board_corners_camera, dtype=np.float64).reshape(4, 2).copy()
        self._lock_frozen = True
        print("board lock: seeded from calibration (frozen until sustained jump)")

    def _apply_board_corners(
        self,
        board_cam: np.ndarray,
        *,
        force_square: bool = True,
        update_projector: bool = True,
        persist: bool = False,
    ) -> None:
        """Update table↔camera (+ optional projector) from a board quad.

        Projector pose is fit so table board corners land on H_c2p(vision corners),
        and the same projector quad is kept for a direct pixel outline.
        Does not rewrite cal on disk unless ``persist`` (avoids hand-wave “recal”).
        """
        cal = self._calibration
        if cal is None:
            return
        board_cam = np.asarray(board_cam, dtype=np.float64).reshape(4, 2)
        if force_square:
            board_cam = force_square_quad(board_cam)
        table = self.layout.board_corners()
        H_t2c = estimate_homography(table, board_cam)
        if H_t2c is None:
            return
        cal.board_corners_camera = board_cam
        cal.H_table_to_camera = H_t2c
        self._last_good_quad = np.array(board_cam, copy=True)
        if update_projector:
            board_proj = warp_points(board_cam, cal.H_camera_to_projector)
            H_t2p = estimate_homography(table, board_proj)
            if H_t2p is None:
                H_t2p = cal.H_camera_to_projector @ H_t2c
            cal.H_table_to_projector = H_t2p
            self._lock_border_proj_px = board_proj
            renderer = getattr(self.runtime, "renderer", None)
            if renderer is not None and hasattr(renderer, "set_homography"):
                renderer.set_homography(H_t2p)
            else:
                print("board lock: renderer missing — border may not move")
            self._lock_dbg_i = getattr(self, "_lock_dbg_i", 0) + 1
            if self._lock_dbg_i <= 2 or self._lock_dbg_i % 20 == 0:
                cam_w = float(np.linalg.norm(board_cam[1] - board_cam[0]))
                cam_h = float(np.linalg.norm(board_cam[3] - board_cam[0]))
                proj_w = float(np.linalg.norm(board_proj[1] - board_proj[0]))
                proj_h = float(np.linalg.norm(board_proj[3] - board_proj[0]))
                pw, ph = cal.projector_size
                cw, ch = cal.camera_size
                print(
                    f"board lock debug: cam_quad≈{cam_w:.0f}x{cam_h:.0f}px "
                    f"(frame {cw}x{ch})  proj_quad≈{proj_w:.0f}x{proj_h:.0f}px "
                    f"(frame {pw}x{ph})  renderer={'yes' if renderer else 'NO'}"
                )
        if persist:
            path = getattr(self.runtime.config, "calibration_path", None)
            if path is not None:
                try:
                    cal.save(path)
                except Exception as exc:  # noqa: BLE001
                    print(f"could not save updated board corners: {exc}")

    def _capture_empty_reference(self, frame_bgr: np.ndarray) -> None:
        roi = self._board_roi()
        if roi is None:
            return
        warped = warp_board(frame_bgr, roi, out_size=400)
        self.occupancy.set_empty_reference_from_warped(warped)
        print("occupancy: empty-square reference captured (board ROI only)")

    def _snapshot_board_gray(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        roi = self._board_roi()
        if roi is None:
            return None
        warped = warp_board(frame_bgr, roi, out_size=400)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

    def _hand_is_moving(self, frame_bgr: np.ndarray) -> bool:
        """Pause classification while a hand is in frame. Keep the pending count."""
        gray = self._snapshot_board_gray(frame_bgr)
        if gray is None:
            return False
        if self._last_gray is None:
            self._last_gray = gray
            return False
        motion = float(np.mean(cv2.absdiff(self._last_gray, gray)))
        self._last_gray = gray
        return motion > 5.0

    def _lock_physical_board(
        self,
        frame_bgr: np.ndarray,
        *,
        update_projector: bool = True,
    ) -> np.ndarray | None:
        """YOLO AABB + classical corner refine → playing-field quad in camera px.

        Once locked, ignore large jumps (hand / false chessboard) unless the new
        quad is stable for several consecutive polls — stops magenta “recal” flaps.
        """
        try:
            from gamebot.board_detect import detect_physical_board_quad
        except Exception as exc:  # noqa: BLE001
            print(f"board lock: unavailable ({exc})")
            return self._last_good_quad
        try:
            # Chessboard only — wood_contour flaps under projection / cyan table.
            result = detect_physical_board_quad(frame_bgr, conf=0.05, allow_wood=False)
        except Exception as exc:  # noqa: BLE001
            print(f"board lock: failed ({exc})")
            self._lock_misses += 1
            return self._last_good_quad if self._lock_misses < 12 else None
        if result is None:
            self._lock_misses += 1
            if self._last_good_quad is not None and self._lock_misses < 12:
                return self._last_good_quad
            if self._lock_misses % 15 == 1:
                print("board lock: no board found")
            return None

        self._lock_misses = 0
        new_quad = np.asarray(result.quad, dtype=np.float64).reshape(4, 2)

        if self._last_good_quad is not None:
            delta = float(np.mean(np.linalg.norm(new_quad - self._last_good_quad, axis=1)))
            # Tiny jitter — keep frozen pose.
            if delta < 10.0:
                self._lock_pending_quad = None
                self._lock_pending_count = 0
                return self._last_good_quad
            # Big jump (hand, wrong AABB half-frame) — require sustained agreement.
            if delta > 28.0 or self._lock_frozen:
                if self._lock_pending_quad is not None:
                    pend_d = float(
                        np.mean(np.linalg.norm(new_quad - self._lock_pending_quad, axis=1))
                    )
                else:
                    pend_d = 999.0
                if pend_d < 12.0:
                    self._lock_pending_count += 1
                else:
                    self._lock_pending_quad = new_quad.copy()
                    self._lock_pending_count = 1
                need = 8 if self._lock_frozen else 4
                if self._lock_pending_count < need:
                    if self._lock_pending_count == 1:
                        print(
                            f"board lock: ignoring jump Δ≈{delta:.0f}px "
                            f"(need {need} stable frames)"
                        )
                    return self._last_good_quad
                print(f"board lock: accepting sustained move Δ≈{delta:.0f}px")
                self._lock_frozen = True
                self._lock_pending_quad = None
                self._lock_pending_count = 0

        aabb = result.aabb
        if result.method != self._last_good_method:
            print(
                f"board lock: {result.method}  aabb=({aabb[0]},{aabb[1]})-({aabb[2]},{aabb[3]}) "
                f"frame={frame_bgr.shape[1]}x{frame_bgr.shape[0]}"
            )
        self._apply_board_corners(
            new_quad, force_square=False, update_projector=update_projector, persist=False
        )
        self._last_good_quad = new_quad.copy()
        self._last_good_method = result.method
        self._last_lock_method = result.method
        self._lock_frozen = True
        return new_quad

    def _resolve_setup_from_camera(self, frame: np.ndarray) -> bool:
        """Lock physical board (YOLO+mech), then corner-piece colour / optional pose."""
        cal = self._calibration
        if cal is None:
            print("setup: no calibration — assuming human=white")
            self.session.human_is_white = True
            return True

        locked = self._lock_physical_board(frame)
        roi = locked if locked is not None else self._board_roi()
        if roi is None:
            roi = cal.board_corners_camera

        # Prefer YOLO white/black counts on each side of the board.
        occ = self.occupancy.detect(frame, roi, prefer_yolo=True)
        if occ.colors is not None:
            found = detect_human_is_white_from_colors(occ.colors)
            if found is not None:
                self.session.human_is_white = found
                print(
                    f"side detect (yolo): you={'white' if found else 'black'} "
                    f"(bot={'black' if found else 'white'})"
                )
                self._capture_empty_reference(frame)
                return found

        setup = setup_from_corner_pieces(frame, search_roi=roi, existing_board=roi)
        if setup is None:
            setup = setup_from_corner_pieces(frame, search_roi=None, existing_board=roi)
        if setup is not None:
            self._apply_board_corners(
                setup.board_corners_px, force_square=False, update_projector=False, persist=False
            )
            self.session.human_is_white = setup.human_is_white
            black_s = (
                f" black@({setup.black_xy[0]:.0f},{setup.black_xy[1]:.0f})"
                if setup.black_xy is not None
                else " (white corner only)"
            )
            print(
                f"corner piece: white@({setup.white_xy[0]:.0f},{setup.white_xy[1]:.0f})"
                f"{black_s} → bot={'white' if setup.bot_is_white else 'black'} "
                f"you={'white' if setup.human_is_white else 'black'}"
            )
            self._capture_empty_reference(frame)
            return setup.human_is_white

        found = detect_human_is_white(frame, roi)
        if found is None:
            print("setup: side inconclusive — leaving quick-start colour; check piece colours")
            self._capture_empty_reference(frame)
            return bool(self.session.human_is_white)
        self.session.human_is_white = found
        print(
            f"side detect: you={'white' if found else 'black'} "
            f"(bot={'black' if found else 'white'})"
        )
        self._capture_empty_reference(frame)
        return found

    def on_enter(self) -> None:
        self.engine.open()
        elo = getattr(self.runtime.config, "quick_start_elo", None)
        if elo is not None:
            guest = next((p for p in self.db.list_profiles() if p.name == "Guest"), None)
            if guest is None:
                guest = self.db.create_profile("Guest", "beginner", 1)
            band, level = elo_to_skill(int(elo))
            print(
                f"quick start: Guest @ ~{elo} Elo → {band.value} level {level} "
                f"— set up the starting position to begin"
            )
            self.session.quick_start(guest, band, level, human_is_white=True)
            self._setup_ready_frames = 0
            self._setup_tick = 0
            self._awaiting_settle = False
            self._reset_pending()
            self._lock_frozen = False
            self._lock_pending_quad = None
            self._lock_pending_count = 0
            self._seed_lock_from_calibration()
            return
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
        if s.phase == Phase.WAIT_SETUP:
            sc = Scene(layout)
            sc.project_chessboard = False
            # Only the vision poly (magenta) — avoid drawing the old cal-H border
            # on top, which looked like a "stuck" small square.
            sc.project_board_border = False
            if self._lock_border_proj_px is not None:
                sc.add_poly_px(self._lock_border_proj_px, (255, 0, 255), 6)
            sc.title = "Set up pieces"
            sc.add_text(
                "Starting position → game starts",
                layout.margin_mm,
                layout.margin_mm + 36,
                (220, 220, 80),
            )
            return sc
        if s.phase == Phase.PROMOTE:
            return build_promotion_scene(layout)
        if s.phase == Phase.FAULT:
            revert = None
            reason = "illegal"
            if s.revert_hint is not None:
                revert = (s.revert_hint.from_sq, s.revert_hint.to_sq)
                reason = s.revert_hint.reason
            sc = build_fault_scene(layout, revert, reason)
            if self._lock_border_proj_px is not None:
                sc.add_poly_px(self._lock_border_proj_px, (0, 0, 255), 6)
            return sc
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
        # Hide bot arrow once the piece should already be placed (human's turn),
        # or as soon as its origin square is empty.
        engine_arrow = None if s.is_human_turn() else s.last_engine_move
        if self._awaiting_settle and not self._arrow_hidden:
            engine_arrow = s.last_engine_move
        elif self._arrow_hidden:
            engine_arrow = None
        scene = build_play_overlay(layout, engine_arrow, checks, revert, promo_label)
        # No filled chess grid during play (blackproj vision); keep outline for alignment.
        scene.project_chessboard = False
        scene.project_board_border = False
        if self._lock_border_proj_px is not None:
            scene.add_poly_px(self._lock_border_proj_px, (255, 0, 255), 5)
        if self._awaiting_settle:
            scene.add_text(
                "Place bot piece (or play your reply too)…",
                layout.margin_mm,
                layout.margin_mm - 28,
                (200, 120, 255),
            )
        elif s.is_human_turn():
            scene.add_text("Your move", layout.margin_mm, layout.margin_mm - 28, (0, 220, 200))
        else:
            scene.add_text("Bot thinking…", layout.margin_mm, layout.margin_mm - 28, (200, 120, 255))
        return scene

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
            self._setup_ready_frames = 0
            self._setup_tick = 0
            self._awaiting_settle = False
            self._reset_pending()
            print("waiting for starting position…")
            self._lock_frozen = False
            self._lock_pending_quad = None
            self._lock_pending_count = 0
            self._seed_lock_from_calibration()
            return
        if zone_id.startswith("promo:"):
            if s.confirm_promotion(int(zone_id.split(":", 1)[1])) and s.board.move_stack:
                self._pending_still = ("human", s.board.move_stack[-1].uci())
            self._awaiting_settle = True
            self._reset_pending()
            return
        if zone_id == "again":
            s.phase = Phase.PROFILE
            self._set_led(LedMode.IDLE)
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
        if s.phase == Phase.WAIT_SETUP:
            self._poll_wait_setup(frame_bgr)
            return
        if s.phase == Phase.FAULT:
            self._set_led(LedMode.ERROR)
            self._poll_fault_recover(frame_bgr)
            self._sync_turn_led()
            return
        if s.phase != Phase.PLAY:
            return
        if self._pending_still is not None:
            who, uci = self._pending_still
            self._pending_still = None
            self._save_move_still(frame_bgr, who=who, uci=uci)
        self._sync_turn_led()
        if self._awaiting_settle or s.is_human_turn():
            self._poll_board_move(frame_bgr)
            self._sync_turn_led()
        if (
            s.phase == Phase.PLAY
            and not s.is_human_turn()
            and not self._waiting_engine
            and not self._awaiting_settle
        ):
            self._waiting_engine = True
            self._do_engine_turn()
            self._waiting_engine = False
            self._sync_turn_led()

    def _poll_wait_setup(self, frame_bgr: np.ndarray) -> None:
        """Stable border from cal; wait until occupancy looks like the starting position."""
        cal = self._calibration
        if cal is None:
            return
        self._setup_tick += 1
        if self._setup_tick == 1 and self._last_good_quad is None:
            self._seed_lock_from_calibration()
        # Rarely re-check vision; sticky lock ignores hand-wave jumps.
        if self._setup_tick % (self._setup_lock_every * 3) == 0:
            self._lock_physical_board(frame_bgr, update_projector=True)

        roi = self._last_good_quad if self._last_good_quad is not None else cal.board_corners_camera
        occ = self.occupancy.detect(frame_bgr, roi, prefer_yolo=True)
        ready = looks_like_starting_occupancy(occ.grid)
        summary = occupancy_debug_summary(occ.grid, method=occ.method, colors=occ.colors)
        # Regular piece-count debug so we can see why setup is stuck.
        if self._setup_tick == 1 or self._setup_tick % 20 == 0 or (ready and self._setup_ready_frames == 0):
            print(f"setup occupancy: {summary}")
        if ready:
            self._setup_ready_frames += 1
            if self._setup_ready_frames == 1 or self._setup_ready_frames % 4 == 0:
                print(f"setup: start-like — holding {self._setup_ready_frames}/8 ({summary})")
        else:
            # Soft decay — brief YOLO flaps (30↔32) shouldn't restart the whole wait.
            self._setup_ready_frames = max(0, self._setup_ready_frames - 2)
            return
        if self._setup_ready_frames < 8:
            return

        human_white = self._resolve_setup_from_camera(frame_bgr)
        side = "white" if human_white else "black"
        print(f"setup ready — you={side}; starting game")
        self.session.confirm_setup(human_is_white=human_white)
        # Persist corners once when the game actually starts.
        if self._last_good_quad is not None:
            self._apply_board_corners(
                self._last_good_quad, force_square=False, update_projector=True, persist=True
            )
        self._awaiting_settle = False
        self._reset_pending()
        self._setup_ready_frames = 0
        self._sync_turn_led()
        self._log_state("game_start", side=side)

    def _do_engine_turn(self) -> None:
        s = self.session
        if s.phase != Phase.PLAY or s.is_human_turn():
            return
        if self._awaiting_settle:
            return
        thinking = getattr(self.runtime, "thinking", None)
        if callable(thinking):
            thinking(True)
        applied = False
        try:
            print(
                f"engine move… stockfish={'yes' if self.engine.available else 'fallback'} "
                f"skill={s.skill_level}"
            )
            result = self.engine.play(s.board, s.skill_level)
            applied = s.apply_engine_move(result.move)
            if not applied:
                print(f"engine move rejected: {result.move.uci()}")
                self._log_state("engine_rejected", uci=result.move.uci())
            else:
                print(
                    f"engine played {result.move.uci()} — place along the arrow "
                    "(you can also play your reply in the same motion)"
                )
                cam = self._camera
                if isinstance(cam, MockCamera):
                    cam.set_occupancy(occupancy_from_board(s.board))
                self._awaiting_settle = True
                self._arrow_hidden = False
                self._reset_pending()
                self._log_state("engine_played", uci=result.move.uci())
        except Exception as exc:  # noqa: BLE001 — keep session alive
            print(f"engine turn failed: {exc}")
            self._log_state("engine_error", error=str(exc))
        finally:
            if callable(thinking):
                thinking(False)
            # Arrow is the human's cue — turquoise as soon as it is on the table.
            self._set_led(LedMode.PLAYER_TURN if applied else LedMode.BOT_TURN)

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
        self._awaiting_settle = True
        self._reset_pending()

    def _hand_to_player_after_bot(self, frame_bgr: np.ndarray) -> None:
        s = self.session
        uci = s.last_engine_move.uci() if s.last_engine_move else "settle"
        self._awaiting_settle = False
        s.last_engine_move = None
        s.last_engine_promotion = None
        self._reset_pending()
        print("board settled — your move")
        self._set_led(LedMode.PLAYER_TURN)
        self._log_state("settle_player_turn")
        self._save_move_still(frame_bgr, who="bot", uci=uci)

    def _apply_confirmed_human(self, move: chess.Move, frame_bgr: np.ndarray, *, source: str) -> None:
        s = self.session
        uci = move.uci()
        if s.try_human_move(move):
            print(f"human move detected ({source}): {uci}")
            self._awaiting_settle = False
            s.last_engine_move = None
            s.last_engine_promotion = None
            self._reset_pending()
            self._sync_turn_led()
            self._log_state("human_move", uci=uci, source=source)
            if s.phase == Phase.PLAY:
                applied = s.board.move_stack[-1].uci() if s.board.move_stack else uci
                self._save_move_still(frame_bgr, who="human", uci=applied)
            cam = self._camera
            if isinstance(cam, MockCamera):
                cam.set_occupancy(occupancy_from_board(s.board))
        elif s.phase == Phase.FAULT:
            squares = [move.from_square, move.to_square]
            self._enter_fault(s.revert_hint.reason if s.revert_hint else "illegal", squares)

    def _confirm_legal(self, move: chess.Move, *, label: str) -> bool:
        """Accumulate still-frame confirms for the same UCI. True when ready to apply."""
        uci = move.uci()
        self._pending_fault_key = None
        self._pending_fault_count = 0
        if uci == self._pending_uci:
            self._pending_count += 1
        else:
            self._pending_uci = uci
            self._pending_count = 1
            print(f"occ: candidate {uci} ({label})")
            self._log_state("occ_candidate", uci=uci, label=label, mismatch=None)
            self._set_led(LedMode.BOT_TURN)
        return self._pending_count >= self._confirm_need

    def _note_engine_piece_left(self, grid: np.ndarray) -> None:
        """Drop the arrow once the bot piece is no longer on its origin square."""
        move = self.session.last_engine_move
        if move is None or not self._awaiting_settle or self._arrow_hidden:
            return
        r, c = square_rc(move.from_square)
        if bool(grid[r, c]):
            return
        self._arrow_hidden = True
        print(f"arrow off — {chess.square_name(move.from_square)} empty")

    def _confirm_fault(self, squares: list[chess.Square], reason: str) -> bool:
        # Square list jitters when YOLO flickers. Latch on the reason, not the squares.
        key = reason
        self._pending_uci = None
        self._pending_count = 0
        if key == self._pending_fault_key:
            self._pending_fault_count += 1
        else:
            self._pending_fault_key = key
            self._pending_fault_count = 1
            print(f"occ: odd board ({reason}) — chilling…")
            self._log_state(
                "occ_fault_candidate",
                reason=reason,
                changed=[chess.square_name(sq) for sq in squares],
            )
        return self._pending_fault_count >= self._fault_confirm_need

    def _observe_board(self, frame_bgr: np.ndarray, board: chess.Board):
        """One still frame → occupancy + verdict. None while a hand is moving."""
        if self._hand_is_moving(frame_bgr):
            return None
        occ = self._read_occupancy(frame_bgr)
        if occ is None:
            return None
        verdict = classify_occupancy_vs_board(
            occ.grid, board, match_slop=0, colors=occ.colors
        )
        return occ, verdict

    def _physical_board_is_pre_engine(self, grid: np.ndarray) -> bool:
        """Pieces still show the position from before the arrow was projected."""
        s = self.session
        if s.last_engine_move is None or not s.board.move_stack:
            return False
        pre = s.board.copy()
        pre.pop()
        return occupancy_mismatch(grid, occupancy_from_board(pre)) <= 1

    def _poll_fault_recover(self, frame_bgr: np.ndarray) -> None:
        """Stay in FAULT until the real position is back for several frames."""
        s = self.session
        if s.phase != Phase.FAULT:
            return
        self._dbg_tick += 1
        observed = self._observe_board(frame_bgr, s.board)
        if observed is None:
            return
        occ, verdict = observed
        if occupancy_frame_is_noise(occ.grid, s.board, verdict):
            return
        if verdict.kind == OccupancyKind.MATCH:
            self._fault_clear_count += 1
            if self._fault_clear_count >= self._fault_clear_need:
                self._clear_fault_state()
            return
        self._fault_clear_count = 0
        if verdict.kind == OccupancyKind.ONE_LEGAL and verdict.move is not None:
            if self._confirm_legal(verdict.move, label="fault_recover"):
                s.phase = Phase.PLAY
                self._apply_confirmed_human(verdict.move, frame_bgr, source="fault_recover")
            return
        if self._dbg_tick % 40 == 0:
            print(
                f"fault: still odd (mismatch={verdict.mismatch} method={occ.method}) "
                "— fix the board…"
            )
            self._log_state(
                "fault_waiting",
                mismatch=verdict.mismatch,
                kind=verdict.kind.value,
                method=occ.method,
            )

    def _poll_board_move(self, frame_bgr: np.ndarray) -> None:
        """Occupancy vs expected FEN: match / one legal / ignore detector noise."""
        s = self.session
        if self._calibration is None or s.phase != Phase.PLAY:
            return
        self._dbg_tick += 1
        observed = self._observe_board(frame_bgr, s.board)
        if observed is None:
            return
        occ, verdict = observed

        if self._awaiting_settle:
            if not s.is_human_turn():
                return
            # Bot just moved in software. Until the pieces leave the previous
            # position, a noisy grid must not become "the human replied".
            if occupancy_frame_is_noise(occ.grid, s.board, verdict):
                return
            self._note_engine_piece_left(occ.grid)
            if self._physical_board_is_pre_engine(occ.grid):
                self._reset_pending()
                return
            if self._dbg_tick % 30 == 0:
                print(
                    f"settle occ: {verdict.kind.value} mismatch={verdict.mismatch} "
                    f"method={occ.method}"
                )
            if verdict.kind == OccupancyKind.MATCH:
                self._pending_fault_key = None
                self._pending_fault_count = 0
                if self._pending_uci != "__match__":
                    self._pending_uci = "__match__"
                    self._pending_count = 1
                    print("occ: board matches engine position — confirming bot place…")
                else:
                    self._pending_count += 1
                if self._pending_count >= self._confirm_need:
                    self._hand_to_player_after_bot(frame_bgr)
                return
            if verdict.kind == OccupancyKind.ONE_LEGAL and verdict.move is not None:
                if self._confirm_legal(verdict.move, label="settle_combo"):
                    self._apply_confirmed_human(verdict.move, frame_bgr, source="settle_combo")
                return
            if verdict.mismatch >= 3 and self._confirm_fault(
                verdict.changed_squares, "unexpected_while_placing"
            ):
                self._enter_fault("unexpected while placing bot piece", verdict.changed_squares)
            elif self._dbg_tick % 40 == 0:
                self._log_state(
                    "settle_wait",
                    kind=verdict.kind.value,
                    mismatch=verdict.mismatch,
                    method=occ.method,
                )
            return

        if not s.is_human_turn():
            return

        if occupancy_frame_is_noise(occ.grid, s.board, verdict):
            return

        if verdict.kind == OccupancyKind.MATCH:
            self._reset_pending()
            return

        changed_names = [chess.square_name(sq) for sq in verdict.changed_squares]
        if verdict.kind == OccupancyKind.ONE_LEGAL and verdict.move is not None:
            if self._confirm_legal(verdict.move, label="human"):
                self._apply_confirmed_human(verdict.move, frame_bgr, source="occupancy")
            else:
                self._log_state(
                    "occ_pending",
                    uci=verdict.move.uci(),
                    mismatch=verdict.mismatch,
                    changed=changed_names,
                    method=occ.method,
                    count=self._pending_count,
                )
            return

        if self._confirm_fault(verdict.changed_squares, "unexpected"):
            self._enter_fault("unexpected board change", verdict.changed_squares)
        else:
            self._log_state(
                "occ_odd",
                mismatch=verdict.mismatch,
                changed=changed_names,
                method=occ.method,
                count=self._pending_fault_count,
            )


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
