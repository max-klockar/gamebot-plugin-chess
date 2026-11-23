from __future__ import annotations

import chess
import cv2
import numpy as np

from chess_plugin.db import Database
from chess_plugin.engine import ChessEngine
from chess_plugin.game import BAND_LEVEL_RANGES, ChessSession, Phase, SkillBand, elo_to_skill
from chess_plugin.ui import (
    build_band_scene,
    build_level_scene,
    build_play_overlay,
    build_profile_scene,
    build_promotion_scene,
    load_icon,
)
from chess_plugin.vision import (
    BoardOccupancyDetector,
    detect_human_is_white,
    hypothesize_moves_from_diff,
    looks_like_starting_occupancy,
    occupancy_debug_summary,
    occupancy_from_board,
    setup_from_corner_pieces,
    square_diff_energy,
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
        self._prev_warped: np.ndarray | None = None
        self._last_gray: np.ndarray | None = None
        self._waiting_engine = False
        self._awaiting_settle = False
        self._settle_frames = 0
        self._pending_uci: str | None = None
        self._pending_count = 0
        self._dbg_tick = 0
        self._exit = False
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
        cal = self._calibration
        if cal is None:
            return
        warped = warp_board(frame_bgr, cal.board_corners_camera, out_size=400)
        self.occupancy.set_empty_reference_from_warped(warped)
        print("occupancy: empty-square reference captured (board ROI only)")

    def _snapshot_board_gray(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        cal = self._calibration
        if cal is None:
            return None
        warped = warp_board(frame_bgr, cal.board_corners_camera, out_size=400)
        return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

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

    def _resolve_setup_from_camera(self) -> bool:
        """Lock physical board (YOLO+mech), then corner-piece colour / optional pose."""
        cal = self._calibration
        cam = self._camera
        if cal is None or cam is None:
            print("setup: no camera/cal — assuming human=white")
            self.session.human_is_white = True
            return True
        frame = cam.read()
        for _ in range(3):
            frame = cam.read()

        locked = self._lock_physical_board(frame)
        roi = locked if locked is not None else cal.board_corners_camera
        setup = setup_from_corner_pieces(frame, search_roi=roi, existing_board=roi)
        if setup is None:
            setup = setup_from_corner_pieces(frame, search_roi=None, existing_board=roi)
        if setup is not None:
            # Keep sticky cal lock for geometry; only take colour/side from corners.
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
            self._prev_warped = self._snapshot_board_gray(frame)
            return setup.human_is_white

        found = detect_human_is_white(frame, cal.board_corners_camera)
        if found is None:
            # Do NOT default to human=white — that is what assigned white to the
            # player while the white army sat on the bot side (see Pi log).
            print("setup: side inconclusive — leaving quick-start colour; check piece colours")
            self._capture_empty_reference(frame)
            self._prev_warped = self._snapshot_board_gray(frame)
            return bool(self.session.human_is_white)
        self.session.human_is_white = found
        print(
            f"side detect: you={'white' if found else 'black'} "
            f"(bot={'black' if found else 'white'})"
        )
        self._capture_empty_reference(frame)
        self._prev_warped = self._snapshot_board_gray(frame)
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
            self._settle_frames = 0
            self._pending_uci = None
            self._pending_count = 0
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
        scene = build_play_overlay(layout, s.last_engine_move, checks, revert, promo_label)
        # No filled chess grid during play (blackproj vision); keep outline for alignment.
        scene.project_chessboard = False
        scene.project_board_border = False
        if self._lock_border_proj_px is not None:
            scene.add_poly_px(self._lock_border_proj_px, (255, 0, 255), 5)
        if s.is_human_turn():
            scene.add_text("Your move", layout.margin_mm, layout.margin_mm - 28, (220, 220, 80))
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
            self._pending_uci = None
            self._pending_count = 0
            print("waiting for starting position…")
            self._lock_frozen = False
            self._lock_pending_quad = None
            self._lock_pending_count = 0
            self._seed_lock_from_calibration()
            return
        if zone_id.startswith("promo:"):
            s.confirm_promotion(int(zone_id.split(":", 1)[1]))
            self._awaiting_settle = True
            self._settle_frames = 0
            self._pending_uci = None
            self._pending_count = 0
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
        if s.phase == Phase.WAIT_SETUP:
            self._poll_wait_setup(frame_bgr)
            return
        if s.phase != Phase.PLAY:
            return
        if self._awaiting_settle or s.is_human_turn():
            self._poll_board_move(frame_bgr)
        if not s.is_human_turn() and not self._waiting_engine and not self._awaiting_settle:
            self._waiting_engine = True
            self._do_engine_turn()
            self._waiting_engine = False

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
        occ = self.occupancy.detect(frame_bgr, roi)
        ready = looks_like_starting_occupancy(occ.grid)
        # Regular piece-count debug so we can see why setup is stuck.
        if self._setup_tick == 1 or self._setup_tick % 20 == 0 or (ready and self._setup_ready_frames == 0):
            print(f"setup occupancy: {occupancy_debug_summary(occ.grid)}")
        if ready:
            self._setup_ready_frames += 1
            if self._setup_ready_frames == 1 or self._setup_ready_frames % 6 == 0:
                print(
                    f"setup: start-like — holding {self._setup_ready_frames}/12 "
                    f"({occupancy_debug_summary(occ.grid)})"
                )
        else:
            self._setup_ready_frames = 0
            return
        if self._setup_ready_frames < 12:
            return

        human_white = self._resolve_setup_from_camera()
        side = "white" if human_white else "black"
        print(f"setup ready — you={side}; starting game")
        self.session.confirm_setup(human_is_white=human_white)
        # Persist corners once when the game actually starts.
        if self._last_good_quad is not None:
            self._apply_board_corners(
                self._last_good_quad, force_square=False, update_projector=True, persist=True
            )
        self._awaiting_settle = False
        self._settle_frames = 0
        self._pending_uci = None
        self._pending_count = 0
        self._setup_ready_frames = 0

    def _do_engine_turn(self) -> None:
        s = self.session
        if s.phase != Phase.PLAY or s.is_human_turn():
            return
        thinking = getattr(self.runtime, "thinking", None)
        if callable(thinking):
            thinking(True)
        try:
            print(
                f"engine move… stockfish={'yes' if self.engine.available else 'fallback'} "
                f"skill={s.skill_level}"
            )
            result = self.engine.play(s.board, s.skill_level)
            s.apply_engine_move(result.move)
            print(f"engine played {result.move.uci()} — place that piece, then your move")
            cam = self._camera
            if isinstance(cam, MockCamera):
                cam.set_occupancy(occupancy_from_board(s.board))
            # Wait for hand motion to settle after placing the bot piece (no ideal occupancy sync).
            self._awaiting_settle = True
            self._settle_frames = 0
            self._pending_uci = None
            self._pending_count = 0
        except Exception as exc:  # noqa: BLE001 — keep session alive
            print(f"engine turn failed: {exc}")
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
        self._awaiting_settle = True
        self._settle_frames = 0

    def _poll_board_move(self, frame_bgr: np.ndarray) -> None:
        """Watch only the warped board ROI; detect moves via frame diff vs legal moves."""
        s = self.session
        cal = self._calibration
        if cal is None or s.phase != Phase.PLAY:
            return
        gray = self._snapshot_board_gray(frame_bgr)
        if gray is None:
            return
        self._dbg_tick += 1

        if self._last_gray is None or self._prev_warped is None:
            self._last_gray = gray
            self._prev_warped = gray.copy()
            return

        frame_motion = float(np.mean(cv2.absdiff(self._last_gray, gray)))
        self._last_gray = gray

        # After engine arrow: wait until the hand stops moving, then lock a new baseline.
        if self._awaiting_settle:
            if frame_motion > 4.0:
                self._settle_frames = 0
            else:
                self._settle_frames += 1
            if self._dbg_tick % 40 == 0:
                print(f"waiting for board to settle… motion={frame_motion:.1f}")
            if self._settle_frames >= 10:
                self._awaiting_settle = False
                self._prev_warped = gray.copy()
                self._pending_uci = None
                self._pending_count = 0
                print("board settled — your move" if s.is_human_turn() else "board settled")
            return

        if not s.is_human_turn():
            return

        # Hand still moving — don't commit a move yet.
        if frame_motion > 5.0:
            self._pending_uci = None
            self._pending_count = 0
            return

        energy = square_diff_energy(self._prev_warped, gray)
        hyps = hypothesize_moves_from_diff(energy, s.board, min_square=8.0)
        if not hyps:
            return
        best = hyps[0]
        second = hyps[1].score if len(hyps) > 1 else 0.0
        if best.score < 22.0:
            return
        if second > 0 and best.score < second * 1.25:
            if self._dbg_tick % 30 == 0:
                print(
                    f"move watch: ambiguous {best.move.uci()} ({best.score:.0f}) "
                    f"vs {hyps[1].move.uci()} ({second:.0f})"
                )
            return

        uci = best.move.uci()
        if uci == self._pending_uci:
            self._pending_count += 1
        else:
            self._pending_uci = uci
            self._pending_count = 1
            print(f"move watch: candidate {uci} score={best.score:.0f}")
        if self._pending_count < 5:
            return
        if s.try_human_move(best.move):
            print(f"human move detected: {uci}")
            self._prev_warped = gray.copy()
            self._pending_uci = None
            self._pending_count = 0


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
