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
        grid = np.zeros((8, 8), dtype=bool)
        conf = np.zeros((8, 8), dtype=np.float64)
        sq = warped.shape[0] // 8
        for r in range(8):
            for c in range(8):
                y0, x0 = r * sq, c * sq
                patch = gray[y0 + 4 : y0 + sq - 4, x0 + 4 : x0 + sq - 4]
                if patch.size == 0:
                    continue
                # Pieces create darker/lighter blobs vs empty wood texture
                std = float(patch.std())
                mean = float(patch.mean())
                if empty_reference is not None:
                    ref = empty_reference[r, c]
                    score = abs(mean - ref)
                    occupied = score > 18.0 and std > 8.0
                    conf[r, c] = min(1.0, score / 40.0)
                else:
                    # Heuristic: occupied squares have higher local std
                    occupied = std > 22.0
                    conf[r, c] = min(1.0, std / 40.0)
                grid[r, c] = occupied
        return OccupancyResult(grid=grid, confidence=conf)


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
