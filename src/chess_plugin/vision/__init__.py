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
    warped_bgr: np.ndarray | None = None


@dataclass
class MoveHypothesis:
    move: chess.Move
    score: float


@dataclass
class CornerPieceSetup:
    """White corner piece defines colour; optional black for opposite-corner pose."""

    white_xy: tuple[float, float]
    black_xy: tuple[float, float] | None
    board_corners_px: np.ndarray  # TL,TR,BR,BL
    human_is_white: bool
    bot_is_white: bool


def _largest_blob(
    mask: np.ndarray,
    *,
    min_area: float = 80.0,
    max_area: float = 25000.0,
) -> tuple[tuple[float, float], float] | None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_xy: tuple[float, float] | None = None
    best_area = min_area
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < best_area or area > max_area:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        best_xy = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        best_area = area
    if best_xy is None:
        return None
    return best_xy, best_area


def detect_white_black_pieces(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find one bright (white) and one dark (black) piece blob in the frame/ROI."""
    work = frame_bgr
    offset = np.array([0.0, 0.0])
    if search_roi is not None and len(search_roi) >= 4:
        # Axis-aligned crop around the ROI to ignore room clutter.
        xs = search_roi[:, 0]
        ys = search_roi[:, 1]
        x0 = max(0, int(xs.min()) - 20)
        y0 = max(0, int(ys.min()) - 20)
        x1 = min(frame_bgr.shape[1], int(xs.max()) + 20)
        y1 = min(frame_bgr.shape[0], int(ys.max()) + 20)
        if x1 - x0 < 40 or y1 - y0 < 40:
            return None
        work = frame_bgr[y0:y1, x0:x1]
        offset = np.array([float(x0), float(y0)])

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    # White piece: very bright compact blob
    white_thr = max(170.0, float(np.percentile(blur, 92)))
    white_mask = cv2.threshold(blur, white_thr, 255, cv2.THRESH_BINARY)[1]
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # Black piece: very dark compact blob
    black_thr = min(80.0, float(np.percentile(blur, 12)))
    black_mask = cv2.threshold(blur, black_thr, 255, cv2.THRESH_BINARY_INV)[1]
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    w = _largest_blob(white_mask)
    b = _largest_blob(black_mask)
    if w is None or b is None:
        return None
    wxy = (w[0][0] + offset[0], w[0][1] + offset[1])
    bxy = (b[0][0] + offset[0], b[0][1] + offset[1])
    if float(np.linalg.norm(np.array(wxy) - np.array(bxy))) < 40.0:
        return None
    return wxy, bxy


def board_quad_from_opposite_corners(
    a: tuple[float, float],
    b: tuple[float, float],
) -> np.ndarray:
    """Square board outer quad (TL,TR,BR,BL) from two opposite corner points."""
    a_xy = np.asarray(a, dtype=np.float64)
    b_xy = np.asarray(b, dtype=np.float64)
    mid = 0.5 * (a_xy + b_xy)
    half = 0.5 * (b_xy - a_xy)
    perp = np.array([-half[1], half[0]], dtype=np.float64)
    corners = np.array([a_xy, mid + perp, b_xy, mid - perp], dtype=np.float64)
    # Spatial order to TL,TR,BR,BL
    by_y = corners[np.argsort(corners[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bot = by_y[2:][np.argsort(by_y[2:, 0])]
    return np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float64)


def detect_white_piece(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Brightest compact blob = white piece."""
    pair = detect_white_black_pieces(frame_bgr, search_roi=search_roi)
    if pair is not None:
        return pair[0]
    # White-only path (no black required)
    work = frame_bgr
    offset = np.array([0.0, 0.0])
    if search_roi is not None and len(search_roi) >= 4:
        xs, ys = search_roi[:, 0], search_roi[:, 1]
        x0 = max(0, int(xs.min()) - 20)
        y0 = max(0, int(ys.min()) - 20)
        x1 = min(frame_bgr.shape[1], int(xs.max()) + 20)
        y1 = min(frame_bgr.shape[0], int(ys.max()) + 20)
        if x1 - x0 >= 40 and y1 - y0 >= 40:
            work = frame_bgr[y0:y1, x0:x1]
            offset = np.array([float(x0), float(y0)])
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thr = max(170.0, float(np.percentile(blur, 92)))
    mask = cv2.threshold(blur, thr, 255, cv2.THRESH_BINARY)[1]
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    blob = _largest_blob(mask)
    if blob is None:
        return None
    return (blob[0][0] + offset[0], blob[0][1] + offset[1])


def snap_board_to_white_corner(
    white_xy: tuple[float, float],
    board_corners_px: np.ndarray,
) -> np.ndarray:
    """Translate existing board quad so the nearest corner sits on the white piece."""
    board = np.asarray(board_corners_px, dtype=np.float64).reshape(4, 2)
    white = np.asarray(white_xy, dtype=np.float64)
    dists = np.linalg.norm(board - white, axis=1)
    i = int(np.argmin(dists))
    return board + (white - board[i])


def setup_from_corner_pieces(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
    existing_board: np.ndarray | None = None,
) -> CornerPieceSetup | None:
    """One white corner piece is enough for colour (+ optional snap of board pose).

    White on the near (bottom) side of the board ⇒ bot is white.
    If a black piece is also visible on the opposite corner, use both for size/pose.
    """
    pair = detect_white_black_pieces(frame_bgr, search_roi=search_roi)
    white_xy: tuple[float, float] | None
    black_xy: tuple[float, float] | None
    if pair is not None:
        white_xy, black_xy = pair
        board = board_quad_from_opposite_corners(white_xy, black_xy)
    else:
        white_xy = detect_white_piece(frame_bgr, search_roi=search_roi)
        black_xy = None
        if white_xy is None:
            return None
        if existing_board is None:
            return None
        board = snap_board_to_white_corner(white_xy, existing_board)

    near_mid = 0.5 * (board[2] + board[3])
    far_mid = 0.5 * (board[0] + board[1])
    white = np.asarray(white_xy, dtype=np.float64)
    bot_is_white = float(np.linalg.norm(white - near_mid)) <= float(np.linalg.norm(white - far_mid))
    return CornerPieceSetup(
        white_xy=white_xy,
        black_xy=black_xy,
        board_corners_px=board,
        human_is_white=not bot_is_white,
        bot_is_white=bot_is_white,
    )


def square_diff_energy(prev_gray: np.ndarray, curr_gray: np.ndarray, inset_frac: float = 0.22) -> np.ndarray:
    """Per-square mean abs-diff inside the warped board (8x8)."""
    diff = cv2.absdiff(prev_gray, curr_gray).astype(np.float32)
    h = diff.shape[0]
    sq = h // 8
    inset = max(6, int(sq * inset_frac))
    energy = np.zeros((8, 8), dtype=np.float64)
    for r in range(8):
        for c in range(8):
            patch = diff[r * sq + inset : (r + 1) * sq - inset, c * sq + inset : (c + 1) * sq - inset]
            energy[r, c] = float(patch.mean()) if patch.size else 0.0
    return energy


def hypothesize_moves_from_diff(
    energy: np.ndarray,
    board: chess.Board,
    *,
    min_square: float = 10.0,
) -> list[MoveHypothesis]:
    """Score legal moves by how much their from/to squares changed in the board ROI."""

    def idx(sq: chess.Square) -> tuple[int, int]:
        return 7 - chess.square_rank(sq), chess.square_file(sq)

    scored: list[MoveHypothesis] = []
    for m in board.legal_moves:
        fr, fc = idx(m.from_square)
        tr, tc = idx(m.to_square)
        e_from = float(energy[fr, fc])
        e_to = float(energy[tr, tc])
        if e_from < min_square and e_to < min_square:
            continue
        # Both ends should change for a normal move; captures still light up both.
        score = e_from + e_to
        scored.append(MoveHypothesis(m, score))
    scored.sort(key=lambda h: h.score, reverse=True)
    return scored


class BoardOccupancyDetector:
    """Estimate occupied squares from a top-down board image.

    Always warps to the board quad first (ignores everything outside the frame).
    Prefers an empty-square brightness reference when available.
    """

    def __init__(self, layout: TableLayout | None = None) -> None:
        self.layout = layout or TableLayout()
        self.empty_reference: np.ndarray | None = None

    def set_empty_reference_from_warped(self, warped_bgr: np.ndarray) -> None:
        """Build empty-square baseline from the board ROI only.

        Uses the middle ranks (usually empty in the starting position) to learn
        light/dark square brightness, then fills an 8x8 checker reference.
        """
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sq = warped_bgr.shape[0] // 8
        means = np.zeros((8, 8), dtype=np.float64)
        for r in range(8):
            for c in range(8):
                patch = gray[r * sq + 6 : (r + 1) * sq - 6, c * sq + 6 : (c + 1) * sq - 6]
                means[r, c] = float(patch.mean()) if patch.size else 0.0
        light: list[float] = []
        dark: list[float] = []
        for r in range(2, 6):  # ranks that are empty at start
            for c in range(8):
                if (r + c) % 2 == 0:
                    light.append(float(means[r, c]))
                else:
                    dark.append(float(means[r, c]))
        light_m = float(np.median(light)) if light else float(np.median(means))
        dark_m = float(np.median(dark)) if dark else float(np.median(means))
        ref = np.zeros((8, 8), dtype=np.float64)
        for r in range(8):
            for c in range(8):
                ref[r, c] = light_m if (r + c) % 2 == 0 else dark_m
        self.empty_reference = ref

    def detect(
        self,
        frame_bgr: np.ndarray,
        board_corners_px: np.ndarray,
        empty_reference: np.ndarray | None = None,
    ) -> OccupancyResult:
        """board_corners_px: 4x2 TL, TR, BR, BL in camera pixels — analysis is board-only."""
        warped = warp_board(frame_bgr, board_corners_px, out_size=400)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blur, 50, 140)
        ref = empty_reference if empty_reference is not None else self.empty_reference
        grid = np.zeros((8, 8), dtype=bool)
        conf = np.zeros((8, 8), dtype=np.float64)
        sq = warped.shape[0] // 8
        for r in range(8):
            for c in range(8):
                y0, x0 = r * sq, c * sq
                # Strong inset: only the square interior (cuts border + projector lines)
                inset = max(8, sq // 5)
                patch = gray[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                epatch = edges[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                if patch.size < 16:
                    continue
                std = float(patch.std())
                mean = float(patch.mean())
                edge_density = float(epatch.mean())
                if ref is not None:
                    score = abs(mean - float(ref[r, c]))
                    # Piece vs empty square: brightness shift and/or extra edges
                    occupied = score > 18.0 or (edge_density > 9.0 and std > 10.0)
                    conf[r, c] = min(1.0, max(score / 45.0, edge_density / 22.0))
                else:
                    occupied = edge_density > 10.0 or std > 18.0
                    conf[r, c] = min(1.0, max(edge_density / 22.0, std / 35.0))
                grid[r, c] = occupied
        return OccupancyResult(grid=grid, confidence=conf, warped_bgr=warped)


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
