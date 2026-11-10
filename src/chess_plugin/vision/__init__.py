from __future__ import annotations

from dataclasses import dataclass

import chess
import cv2
import numpy as np

from gamebot.geometry import TableLayout


@dataclass
class OccupancyResult:
    grid: np.ndarray  # 8x8 bool
    confidence: np.ndarray  # 8x8 float


@dataclass
class MoveHypothesis:
    move: chess.Move
    score: float


class BoardOccupancyDetector:
    """Estimate occupied squares from a top-down board image.

    Uses average dark/light deviation vs empty-square baseline when provided;
    otherwise thresholds local contrast inside each square ROI.
    """

    def __init__(self, layout: TableLayout | None = None) -> None:
        self.layout = layout or TableLayout()

    def detect(
        self,
        frame_bgr: np.ndarray,
        board_corners_px: np.ndarray,
        empty_reference: np.ndarray | None = None,
    ) -> OccupancyResult:
        """board_corners_px: 4x2 TL, TR, BR, BL in camera pixels."""
        warped = warp_board(frame_bgr, board_corners_px, out_size=400)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 40, 120)
        grid = np.zeros((8, 8), dtype=bool)
        conf = np.zeros((8, 8), dtype=np.float64)
        sq = warped.shape[0] // 8
        for r in range(8):
            for c in range(8):
                y0, x0 = r * sq, c * sq
                # Inset avoids square borders / projected overlays
                patch = gray[y0 + 6 : y0 + sq - 6, x0 + 6 : x0 + sq - 6]
                epatch = edges[y0 + 6 : y0 + sq - 6, x0 + 6 : x0 + sq - 6]
                if patch.size == 0:
                    continue
                std = float(patch.std())
                mean = float(patch.mean())
                edge_density = float(epatch.mean())  # 0..255-ish after Canny
                if empty_reference is not None:
                    ref = empty_reference[r, c]
                    score = abs(mean - ref)
                    occupied = (score > 14.0 and std > 6.0) or edge_density > 6.0
                    conf[r, c] = min(1.0, max(score / 40.0, edge_density / 20.0))
                else:
                    # Pieces add edges + texture vs flat blue/white squares
                    occupied = edge_density > 7.0 or std > 16.0
                    conf[r, c] = min(1.0, max(edge_density / 20.0, std / 35.0))
                grid[r, c] = occupied
        return OccupancyResult(grid=grid, confidence=conf)


def occupancy_mismatch(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def detect_human_is_white(
    frame_bgr: np.ndarray,
    board_corners_px: np.ndarray,
) -> bool | None:
    """Infer whether the human plays white from the starting armies.

    Warps the board (TL→top-left … BL→bottom-left). The calibrated *near* side
    (rows 6–7) is the bot/machine side of the table; the far side (rows 0–1) is
    the player. Lighter pieces = white. So lighter-on-near → bot is white →
    human_is_white=False. Returns None if inconclusive.
    """
    warped = warp_board(frame_bgr, board_corners_px, out_size=400)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    sq = warped.shape[0] // 8

    def rank_piece_brightness(rows: range) -> list[float]:
        vals: list[float] = []
        for r in rows:
            for c in range(8):
                y0, x0 = r * sq, c * sq
                patch = gray[y0 + 6 : y0 + sq - 6, x0 + 6 : x0 + sq - 6]
                if patch.size < 16:
                    continue
                # Occupied-ish: higher texture than empty blue/white squares alone
                if float(patch.std()) < 12.0:
                    continue
                vals.append(float(patch.mean()))
        return vals

    near = rank_piece_brightness(range(6, 8))  # bot / machine side of table
    far = rank_piece_brightness(range(0, 2))  # human side
    if len(near) < 4 or len(far) < 4:
        return None
    near_m = float(np.mean(near))
    far_m = float(np.mean(far))
    # Need a clear brightness gap between the two armies
    if abs(near_m - far_m) < 12.0:
        return None
    bot_is_white = near_m > far_m
    return not bot_is_white


def warp_board(frame_bgr: np.ndarray, corners_px: np.ndarray, out_size: int = 400) -> np.ndarray:
    src = np.asarray(corners_px, dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, H, (out_size, out_size))


def occupancy_from_board(board: chess.Board) -> np.ndarray:
    """Ground-truth occupancy from a python-chess board (row0 = rank8)."""
    g = np.zeros((8, 8), dtype=bool)
    for sq, _piece in board.piece_map().items():
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        g[7 - rank, file] = True
    return g


def hypothesize_moves(
    before: np.ndarray,
    after: np.ndarray,
    board: chess.Board,
) -> list[MoveHypothesis]:
    """Match occupancy delta to legal moves."""
    vacated = np.argwhere(before & ~after)
    filled = np.argwhere(~before & after)

    def idx_to_square(r: int, c: int) -> chess.Square:
        return chess.square(c, 7 - r)

    candidates: list[MoveHypothesis] = []
    legal = list(board.legal_moves)

    # Standard: one vacated, one filled
    if len(vacated) == 1 and len(filled) == 1:
        fr, fc = vacated[0]
        tr, tc = filled[0]
        src = idx_to_square(int(fr), int(fc))
        dst = idx_to_square(int(tr), int(tc))
        for m in legal:
            if m.from_square == src and m.to_square == dst:
                candidates.append(MoveHypothesis(m, 1.0))

    # Capture: vacated origin, destination stayed occupied
    if len(vacated) == 1 and len(filled) == 0:
        fr, fc = vacated[0]
        src = idx_to_square(int(fr), int(fc))
        for m in legal:
            if m.from_square == src and board.is_capture(m):
                # Destination should still be occupied
                dr = 7 - chess.square_rank(m.to_square)
                dc = chess.square_file(m.to_square)
                if after[dr, dc]:
                    candidates.append(MoveHypothesis(m, 0.9))

    # Castling: king and rook both move — two vacated, two filled
    if len(vacated) == 2 and len(filled) == 2:
        for m in legal:
            if board.is_castling(m):
                candidates.append(MoveHypothesis(m, 0.95))

    candidates.sort(key=lambda h: h.score, reverse=True)
    # Deduplicate by UCI
    seen: set[str] = set()
    unique: list[MoveHypothesis] = []
    for h in candidates:
        u = h.move.uci()
        if u not in seen:
            seen.add(u)
            unique.append(h)
    return unique


@dataclass
class TokenDetection:
    xy_px: tuple[int, int]
    radius: float
    score: float


class TokenTapDetector:
    """Find a distinct colored token (default red) for projected button presses."""

    def __init__(
        self,
        hsv_low: tuple[int, int, int] = (0, 120, 80),
        hsv_high: tuple[int, int, int] = (10, 255, 255),
        hsv_low2: tuple[int, int, int] = (170, 120, 80),
        hsv_high2: tuple[int, int, int] = (180, 255, 255),
        min_area: int = 80,
    ) -> None:
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.hsv_low2 = np.array(hsv_low2, dtype=np.uint8)
        self.hsv_high2 = np.array(hsv_high2, dtype=np.uint8)
        self.min_area = min_area

    def detect(self, frame_bgr: np.ndarray) -> TokenDetection | None:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_low, self.hsv_high)
        mask2 = cv2.inRange(hsv, self.hsv_low2, self.hsv_high2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: TokenDetection | None = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            score = float(area)
            if best is None or score > best.score:
                best = TokenDetection((int(x), int(y)), float(radius), score)
        return best
