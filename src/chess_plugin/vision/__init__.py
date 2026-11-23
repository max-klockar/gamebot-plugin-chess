from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import chess
import cv2
import numpy as np

from gamebot.geometry import TableLayout, force_square_quad


@dataclass
class OccupancyResult:
    grid: np.ndarray  # 8x8 bool
    confidence: np.ndarray  # 8x8 float
    warped_bgr: np.ndarray | None = None
    method: str = "classical"
    colors: np.ndarray | None = None  # 8x8 int8: -1 empty, 0 white, 1 black (YOLO)


@dataclass
class MoveHypothesis:
    move: chess.Move
    score: float


class OccupancyKind(str, Enum):
    MATCH = "match"
    ONE_LEGAL = "one_legal"
    ILLEGAL_OR_MULTI = "illegal_or_multi"


@dataclass
class OccupancyVerdict:
    kind: OccupancyKind
    mismatch: int
    move: chess.Move | None = None
    changed_squares: list[chess.Square] = field(default_factory=list)


@dataclass
class CornerPieceSetup:
    """White corner piece defines colour; optional black for opposite-corner pose."""

    white_xy: tuple[float, float]
    black_xy: tuple[float, float] | None
    board_corners_px: np.ndarray  # TL,TR,BR,BL
    human_is_white: bool
    bot_is_white: bool


@dataclass
class PieceBlob:
    xy: tuple[float, float]
    area: float
    score: float
    mean_gray: float


def _crop_roi(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None,
    pad: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (crop, offset_xy)."""
    if search_roi is None or len(search_roi) < 4:
        return frame_bgr, np.array([0.0, 0.0])
    xs, ys = search_roi[:, 0], search_roi[:, 1]
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(frame_bgr.shape[1], int(xs.max()) + pad)
    y1 = min(frame_bgr.shape[0], int(ys.max()) + pad)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return frame_bgr, np.array([0.0, 0.0])
    return frame_bgr[y0:y1, x0:x1], np.array([float(x0), float(y0)])


def _nms_blobs(blobs: list[PieceBlob], max_keep: int, min_dist: float) -> list[PieceBlob]:
    """Keep highest-score blobs that are spatially separated (one per piece)."""
    kept: list[PieceBlob] = []
    for b in blobs:
        if len(kept) >= max_keep:
            break
        if any(float(np.linalg.norm(np.array(b.xy) - np.array(k.xy))) < min_dist for k in kept):
            continue
        kept.append(b)
    return kept


def _table_surface_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Largest greenish mid-tone region (play table), dilated; full frame if unsure."""
    b, g, r = cv2.split(frame_bgr)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    raw = (
        (g.astype(np.int16) >= r.astype(np.int16) - 8)
        & (g.astype(np.int16) >= b.astype(np.int16) - 20)
        & (gray >= 75)
        & (gray <= 210)
    ).astype(np.uint8) * 255
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw)
    h, w = gray.shape[:2]
    if n <= 1:
        return np.full((h, w), 255, dtype=np.uint8)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8) * 255
    if int((mask > 0).sum()) < h * w * 0.04:
        return np.full((h, w), 255, dtype=np.uint8)
    # Mild inward shrink drops person/rim clutter; dilate so pieces on the cloth remain.
    core = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    if int((core > 0).sum()) < h * w * 0.02:
        core = mask
    return cv2.dilate(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)))


def _blobs_from_response(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    resp: np.ndarray,
    surface: np.ndarray,
    *,
    thr: int,
    want_bright: bool,
    min_area: float,
    max_area: float,
    min_circ: float = 0.58,
) -> list[PieceBlob]:
    """Round local tophat/blackhat peaks on the table surface."""
    h, w = gray.shape[:2]
    gated = resp.copy()
    gated[surface == 0] = 0
    mask = (gated >= thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # Slight dilate so soft wood tops become one contour, not a speck.
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    b, _g, r = cv2.split(frame_bgr)
    out: list[PieceBlob] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue
        peri = cv2.arcLength(cnt, True)
        if peri < 1e-3:
            continue
        circularity = float(4.0 * np.pi * area / (peri * peri))
        if circularity < min_circ:
            continue
        (_cx, _cy), (bw, bh), _ang = cv2.minAreaRect(cnt)
        if bw < 1 or bh < 1:
            continue
        if max(bw, bh) / min(bw, bh) > 1.7:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            continue
        cx = float(m["m10"] / m["m00"])
        cy = float(m["m01"] / m["m00"])
        if cx < 6 or cy < 6 or cx > w - 6 or cy > h - 6:
            continue
        ix, iy = int(cx), int(cy)
        mean_g = float(gray[iy, ix])
        resp_v = float(gated[iy, ix])
        if want_bright:
            if mean_g < 100:
                continue
            # Cream wood is warm (R>B). Reject cyan projected squares.
            if int(b[iy, ix]) > int(r[iy, ix]) + 22 and mean_g > 165:
                continue
            score = circularity * (mean_g / 255.0) * min(1.0, area / 220.0) * min(1.5, resp_v / 28.0)
        else:
            if mean_g > 120:
                continue
            score = circularity * (1.0 - mean_g / 255.0) * min(1.0, area / 220.0) * min(1.5, resp_v / 45.0)
        out.append(PieceBlob((cx, cy), area, float(score), mean_g))
    out.sort(key=lambda blob: blob.score, reverse=True)
    return out


def find_piece_blobs(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
    *,
    max_each: int = 16,
) -> tuple[list[PieceBlob], list[PieceBlob]]:
    """Wood-on-table piece blobs via local morph contrast (tophat / blackhat)."""
    work, offset = _crop_roi(frame_bgr, search_roi)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    surface = _table_surface_mask(work)
    if search_roi is not None:
        # Cal board ROI crop: trust the crop, don't require green-table mask.
        surface = np.full(gray.shape, 255, dtype=np.uint8)

    # Kernel must exceed piece diameter so the blob center stays a peak (not hollow).
    hat_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
    tophat = cv2.morphologyEx(blur, cv2.MORPH_TOPHAT, hat_k)
    blackhat = cv2.morphologyEx(blur, cv2.MORPH_BLACKHAT, hat_k)

    h, w = gray.shape[:2]
    roi_area = float(h * w)
    # Wood tops are small local peaks; allow a wide area band for filled synthetics too.
    min_a = max(40.0, roi_area * 0.00005)
    max_a = min(2500.0, roi_area * 0.02)
    min_sep = max(26.0, (min_a**0.5) * 1.5)

    whites = _nms_blobs(
        _blobs_from_response(
            work, blur, tophat, surface, thr=12, want_bright=True, min_area=min_a, max_area=max_a
        ),
        max_each,
        min_sep,
    )
    blacks = _nms_blobs(
        _blobs_from_response(
            work, blur, blackhat, surface, thr=24, want_bright=False, min_area=min_a, max_area=max_a
        ),
        max_each,
        min_sep,
    )

    def shift(blobs: list[PieceBlob]) -> list[PieceBlob]:
        return [
            PieceBlob((b.xy[0] + offset[0], b.xy[1] + offset[1]), b.area, b.score, b.mean_gray)
            for b in blobs
        ]

    return shift(whites), shift(blacks)


def detect_white_black_pieces(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Find one bright (white) and one dark (black) piece blob in the frame/ROI."""
    whites, blacks = find_piece_blobs(frame_bgr, search_roi, max_each=1)
    if not whites or not blacks:
        return None
    wxy, bxy = whites[0].xy, blacks[0].xy
    if float(np.linalg.norm(np.array(wxy) - np.array(bxy))) < 50.0:
        return None
    return wxy, bxy


def board_quad_from_opposite_corners(
    a: tuple[float, float],
    b: tuple[float, float],
) -> np.ndarray:
    """Square board outer quad (TL,TR,BR,BL) from two opposite corner points.

    Builds a square on the diagonal a–b (equal sides), then re-asserts the
    shortest-edge square so length/width cannot drift apart.
    """
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
    ordered = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float64)
    return force_square_quad(ordered)


def detect_white_piece(
    frame_bgr: np.ndarray,
    search_roi: np.ndarray | None = None,
) -> tuple[float, float] | None:
    """Best white piece blob (no black required)."""
    whites, _blacks = find_piece_blobs(frame_bgr, search_roi, max_each=1)
    if not whites:
        return None
    return whites[0].xy


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

    Board quad TL,TR,BR,BL: bottom edge (BR–BL) is the bot/machine side of the
    table; top is the human. White nearer the bottom ⇒ bot is white.
    """
    roi = search_roi if search_roi is not None else existing_board
    pair = detect_white_black_pieces(frame_bgr, search_roi=roi)
    white_xy: tuple[float, float] | None
    black_xy: tuple[float, float] | None
    if pair is not None:
        white_xy, black_xy = pair
        board = board_quad_from_opposite_corners(white_xy, black_xy)
    else:
        white_xy = detect_white_piece(frame_bgr, search_roi=roi)
        black_xy = None
        if white_xy is None:
            return None
        if existing_board is None:
            return None
        board = force_square_quad(snap_board_to_white_corner(white_xy, existing_board))

    bot_mid = 0.5 * (board[2] + board[3])  # BR+BL = bottom of camera = bot
    human_mid = 0.5 * (board[0] + board[1])  # TL+TR = top = human
    white = np.asarray(white_xy, dtype=np.float64)
    bot_is_white = float(np.linalg.norm(white - bot_mid)) <= float(np.linalg.norm(white - human_mid))
    return CornerPieceSetup(
        white_xy=white_xy,
        black_xy=black_xy,
        board_corners_px=board,
        human_is_white=not bot_is_white,
        bot_is_white=bot_is_white,
    )


def square_rc(sq: chess.Square) -> tuple[int, int]:
    """Board image row/col for a chess square (rank 8 = row 0)."""
    return 7 - chess.square_rank(sq), chess.square_file(sq)


def squares_affected_by_move(board_before: chess.Board, move: chess.Move) -> list[chess.Square]:
    """Squares that change when `move` is played on `board_before`."""
    sqs = [move.from_square, move.to_square]
    if board_before.is_castling(move):
        if board_before.is_kingside_castling(move):
            rook_from = chess.H1 if board_before.turn == chess.WHITE else chess.H8
            rook_to = chess.F1 if board_before.turn == chess.WHITE else chess.F8
        else:
            rook_from = chess.A1 if board_before.turn == chess.WHITE else chess.A8
            rook_to = chess.D1 if board_before.turn == chess.WHITE else chess.D8
        sqs.extend([rook_from, rook_to])
    elif board_before.is_en_passant(move):
        cap = chess.square(chess.square_file(move.to_square), chess.square_rank(move.from_square))
        sqs.append(cap)
    return sqs


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

    def ensure_empty_reference(self, warped_bgr: np.ndarray) -> bool:
        """Learn empty light/dark means from middle ranks if mid looks clear."""
        if self.empty_reference is not None:
            return False
        gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
        sq = warped_bgr.shape[0] // 8
        # Rough occupancy without a ref — middle should be mostly empty at start.
        busy = 0
        for r in range(2, 6):
            for c in range(8):
                y0, x0 = r * sq, c * sq
                inset = max(6, sq // 6)
                patch = gray[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                if patch.size < 16:
                    continue
                if float(patch.std()) > 16.0 or float(cv2.Laplacian(patch, cv2.CV_64F).var()) > 80.0:
                    busy += 1
        if busy > 6:
            return False
        self.set_empty_reference_from_warped(warped_bgr)
        return True

    def detect(
        self,
        frame_bgr: np.ndarray,
        board_corners_px: np.ndarray,
        empty_reference: np.ndarray | None = None,
        *,
        prefer_yolo: bool = True,
    ) -> OccupancyResult:
        """board_corners_px: 4x2 TL, TR, BR, BL in camera pixels — analysis is board-only.

        When ``pieces.onnx`` is present, YOLO is authoritative — including 0 pieces
        (means “not set up yet”). Classical texture is only used if YOLO weights
        are missing (it over-fires and must not override a real 0).
        """
        if prefer_yolo:
            yolo = occupancy_from_yolo_pieces(frame_bgr, board_corners_px)
            if yolo is not None:
                return yolo
            print("occupancy: pieces.onnx missing — classical fallback (legacy)")
        warped = warp_board(frame_bgr, board_corners_px, out_size=400)
        if empty_reference is None and self.empty_reference is None:
            if self.ensure_empty_reference(warped):
                print("occupancy: learned empty-square ref from clear middle ranks")
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        # Softer Canny — small plastic pieces barely edge under projector light.
        edges = cv2.Canny(blur, 30, 100)
        ref = empty_reference if empty_reference is not None else self.empty_reference
        grid = np.zeros((8, 8), dtype=bool)
        conf = np.zeros((8, 8), dtype=np.float64)
        sq = warped.shape[0] // 8
        for r in range(8):
            for c in range(8):
                y0, x0 = r * sq, c * sq
                # Milder inset — pawns near square edges were getting cropped out.
                inset = max(5, sq // 7)
                patch = gray[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                epatch = edges[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                bgr_patch = warped[y0 + inset : y0 + sq - inset, x0 + inset : x0 + sq - inset]
                if patch.size < 16:
                    continue
                std = float(patch.std())
                mean = float(patch.mean())
                edge_density = float(epatch.mean())
                lap = float(cv2.Laplacian(patch, cv2.CV_64F).var())
                color_std = float(bgr_patch.reshape(-1, 3).std(axis=0).mean()) if bgr_patch.size else 0.0
                texture = max(std, lap / 8.0, color_std)
                if ref is not None:
                    score = abs(mean - float(ref[r, c]))
                    # Dark-on-dark / light-on-light: brightness score alone fails — use texture.
                    occupied = (
                        score > 12.0
                        or edge_density > 5.0
                        or texture > 11.0
                        or lap > 55.0
                    )
                    conf[r, c] = min(
                        1.0,
                        max(score / 40.0, edge_density / 18.0, texture / 28.0, lap / 120.0),
                    )
                else:
                    occupied = edge_density > 6.0 or texture > 12.0 or lap > 60.0
                    conf[r, c] = min(1.0, max(edge_density / 18.0, texture / 28.0, lap / 120.0))
                grid[r, c] = occupied
        return OccupancyResult(
            grid=grid, confidence=conf, warped_bgr=warped, method="classical"
        )


def occupancy_mismatch(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a) != np.asarray(b)))


def detect_human_is_white(
    frame_bgr: np.ndarray,
    board_corners_px: np.ndarray,
) -> bool | None:
    """Infer whether the human plays white from the starting armies.

    Warps TL→top-left … BL→bottom-left. Bottom ranks (rows 6–7) are the
    bot/machine side of the table; top (0–1) is the human. Lighter army = white.
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
                # Occupied-ish: texture / edges (soft — plastic pieces are dull)
                if float(patch.std()) < 8.0 and float(cv2.Laplacian(patch, cv2.CV_64F).var()) < 40.0:
                    continue
                vals.append(float(patch.mean()))
        return vals

    bot_side = rank_piece_brightness(range(6, 8))  # bottom of camera = bot
    human_side = rank_piece_brightness(range(0, 2))  # top = human

    def rank_mean(rows: range) -> float:
        vals: list[float] = []
        for r in rows:
            for c in range(8):
                y0, x0 = r * sq, c * sq
                patch = gray[y0 + 6 : y0 + sq - 6, x0 + 6 : x0 + sq - 6]
                if patch.size:
                    vals.append(float(patch.mean()))
        return float(np.mean(vals)) if vals else 0.0

    if len(human_side) >= 4 and len(bot_side) >= 4:
        human_m = float(np.mean(human_side))
        bot_m = float(np.mean(bot_side))
        source = "pieces"
    else:
        # Whole-rank means still skew toward the white army; better than guessing.
        human_m = rank_mean(range(0, 2))
        bot_m = rank_mean(range(6, 8))
        source = "rank-mean"
        print(
            f"side detect: sparse piece samples "
            f"(human={len(human_side)} bot={len(bot_side)}) — using {source}"
        )
    gap = human_m - bot_m  # >0 → human army brighter → human white
    print(
        f"side detect ({source}): human_side_bright≈{human_m:.0f} "
        f"bot_side_bright≈{bot_m:.0f} gap≈{gap:.0f}"
    )
    if abs(gap) < 4.0:
        return None
    return gap > 0.0


def warp_board(frame_bgr: np.ndarray, corners_px: np.ndarray, out_size: int = 400) -> np.ndarray:
    src = np.asarray(corners_px, dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame_bgr, H, (out_size, out_size))


def camera_xy_to_board_rc(
    xy: tuple[float, float],
    board_corners_px: np.ndarray,
    *,
    out_size: int = 400,
) -> tuple[int, int] | None:
    """Map a camera pixel to board (row, col) with row0=rank8 … row7=rank1."""
    src = np.asarray(board_corners_px, dtype=np.float32).reshape(4, 2)
    dst = np.array(
        [[0, 0], [out_size - 1, 0], [out_size - 1, out_size - 1], [0, out_size - 1]],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    pt = cv2.perspectiveTransform(np.array([[[xy[0], xy[1]]]], dtype=np.float32), H)[0, 0]
    x, y = float(pt[0]), float(pt[1])
    if x < -2 or y < -2 or x > out_size + 1 or y > out_size + 1:
        return None
    sq = out_size / 8.0
    c = int(np.clip(x / sq, 0, 7))
    r = int(np.clip(y / sq, 0, 7))
    return r, c


_yolo_log_key: tuple[int, int, int, int, str] | None = None
_yolo_log_i = 0


def _log_yolo_occupancy(n: int, n_w: int, n_b: int, raw_boxes: int, extra: str) -> None:
    """Print when the piece count changes, plus a slow heartbeat."""
    global _yolo_log_key, _yolo_log_i
    _yolo_log_i += 1
    key = (n, n_w, n_b, raw_boxes, extra)
    if key == _yolo_log_key and _yolo_log_i % 20 != 0:
        return
    _yolo_log_key = key
    print(f"occupancy yolo: {n} on-board (white={n_w} black={n_b}, raw_boxes={raw_boxes}{extra})")


def occupancy_from_yolo_pieces(
    frame_bgr: np.ndarray,
    board_corners_px: np.ndarray,
    *,
    conf: float = 0.15,
) -> OccupancyResult | None:
    """Piece YOLO → 8×8 occupancy (+ white/black colours). None if weights missing."""
    try:
        from gamebot.piece_detect import CLASS_BLACK, CLASS_WHITE, detect_pieces, pieces_available
    except Exception:
        return None
    if not pieces_available():
        return None
    dets = detect_pieces(frame_bgr, conf=conf, search_roi=board_corners_px)
    grid = np.zeros((8, 8), dtype=bool)
    conf_g = np.zeros((8, 8), dtype=np.float64)
    colors = np.full((8, 8), -1, dtype=np.int8)
    off_board = 0
    same_square = 0
    for d in dets:
        rc = camera_xy_to_board_rc(d.xy, board_corners_px)
        if rc is None:
            off_board += 1
            continue
        r, c = rc
        # Keep highest-confidence piece if two map to same square.
        if grid[r, c]:
            same_square += 1
            if conf_g[r, c] >= d.conf:
                continue
        grid[r, c] = True
        conf_g[r, c] = d.conf
        colors[r, c] = CLASS_WHITE if d.cls == CLASS_WHITE else CLASS_BLACK
    warped = warp_board(frame_bgr, board_corners_px, out_size=400)
    n_w = int(np.count_nonzero(colors == CLASS_WHITE))
    n_b = int(np.count_nonzero(colors == CLASS_BLACK))
    extra = ""
    if off_board:
        extra += f", {off_board} off-board"
    if same_square:
        extra += f", {same_square} same-square"
    _log_yolo_occupancy(int(grid.sum()), n_w, n_b, len(dets), extra)
    return OccupancyResult(
        grid=grid, confidence=conf_g, warped_bgr=warped, method="yolo", colors=colors
    )


def detect_human_is_white_from_colors(colors: np.ndarray) -> bool | None:
    """Bottom ranks = bot, top = human. More white pieces on a side ⇒ that side is white."""
    c = np.asarray(colors)
    if c.shape != (8, 8):
        return None
    human = c[0:2]  # top of warp / camera
    bot = c[6:8]
    hw = int(np.count_nonzero(human == 0))
    hb = int(np.count_nonzero(human == 1))
    bw = int(np.count_nonzero(bot == 0))
    bb = int(np.count_nonzero(bot == 1))
    print(f"side detect (yolo): human w/b={hw}/{hb}  bot w/b={bw}/{bb}")
    if hw + hb < 4 or bw + bb < 4:
        return None
    human_white_frac = hw / max(hw + hb, 1)
    bot_white_frac = bw / max(bw + bb, 1)
    if abs(human_white_frac - bot_white_frac) < 0.15:
        return None
    return human_white_frac > bot_white_frac


def occupancy_from_board(board: chess.Board) -> np.ndarray:
    """Ground-truth occupancy from a python-chess board (row0 = rank8)."""
    g = np.zeros((8, 8), dtype=bool)
    for sq, _piece in board.piece_map().items():
        file = chess.square_file(sq)
        rank = chess.square_rank(sq)
        g[7 - rank, file] = True
    return g


def occupancy_debug_summary(
    grid: np.ndarray,
    *,
    method: str | None = None,
    colors: np.ndarray | None = None,
) -> str:
    """One-line debug: piece counts vs starting position (row0=rank8 … row7=rank1)."""
    g = np.asarray(grid, dtype=bool)
    expected = occupancy_from_board(chess.Board())
    n = int(g.sum())
    miss = occupancy_mismatch(g, expected)
    per_rank = [int(g[r].sum()) for r in range(8)]
    # rank labels from white's perspective at bottom of warped board (row7 = rank1)
    rank_bits = " ".join(f"r{8 - r}:{per_rank[r]}" for r in range(8))
    back = per_rank[0] + per_rank[1] + per_rank[6] + per_rank[7]
    mid = sum(per_rank[2:6])
    prefix = f"[{method}] " if method else ""
    colour_bit = ""
    if colors is not None:
        colour_bit = (
            f" w={int(np.count_nonzero(colors == 0))}"
            f" b={int(np.count_nonzero(colors == 1))}"
        )
    return (
        f"{prefix}pieces≈{n}/32{colour_bit} mismatch≈{miss} "
        f"back≈{back}/32 mid≈{mid}/0 | {rank_bits}"
    )


def looks_like_starting_occupancy(
    grid: np.ndarray,
    *,
    max_mismatch: int = 10,
) -> bool:
    """True if occupancy roughly matches the initial chess setup (32 pieces).

    Soft fallback: both armies present on the back ranks and middle clear.
    """
    g = np.asarray(grid, dtype=bool)
    expected = occupancy_from_board(chess.Board())
    if occupancy_mismatch(g, expected) <= max_mismatch:
        return True
    back = int(g[0:2].sum() + g[6:8].sum())
    mid = int(g[2:6].sum())
    # ~24+ of 32 back-rank squares with empty mid is enough to start.
    return back >= 24 and mid <= 8


def _idx_to_square(r: int, c: int) -> chess.Square:
    return chess.square(c, 7 - r)


def changed_squares(before: np.ndarray, after: np.ndarray) -> list[chess.Square]:
    """Squares whose occupied/empty bit differs between two grids."""
    a = np.asarray(before, dtype=bool)
    b = np.asarray(after, dtype=bool)
    out: list[chess.Square] = []
    for r, c in np.argwhere(a != b):
        out.append(_idx_to_square(int(r), int(c)))
    return out


def colors_support_move(
    colors: np.ndarray | None,
    move: chess.Move,
    board: chess.Board,
) -> bool:
    """YOLO colours on the squares this move touches match the resulting position.

    Classical occupancy has no colours and is accepted. A destination that still
    shows the other side is a missed box, not a capture.
    """
    if colors is None:
        return True
    if move not in board.legal_moves:
        # Promotion choice is collapsed to a bare from-to before the piece is picked.
        promo = [
            m
            for m in board.legal_moves
            if m.from_square == move.from_square
            and m.to_square == move.to_square
            and m.promotion
        ]
        if not promo:
            return False
        move = promo[0]
    after = board.copy()
    after.push(move)
    for sq in squares_affected_by_move(board, move):
        r, c = square_rc(sq)
        seen = int(colors[r, c])
        piece = after.piece_at(sq)
        if piece is None:
            if seen != -1:
                return False
        else:
            want = 0 if piece.color == chess.WHITE else 1
            if seen != want:
                return False
    return True


def classify_occupancy_vs_board(
    seen: np.ndarray,
    board: chess.Board,
    *,
    match_slop: int = 0,
    colors: np.ndarray | None = None,
) -> OccupancyVerdict:
    """Compare YOLO/classical occupancy to the software board.

    - match: seen ≈ expected (unchanged)
    - one_legal: delta is exactly one legal move, and colours agree when present
    - illegal_or_multi: anything else (wrong change or too many changes)
    """
    expected = occupancy_from_board(board)
    seen_b = np.asarray(seen, dtype=bool)
    miss = occupancy_mismatch(seen_b, expected)
    changed = changed_squares(expected, seen_b)
    if miss <= match_slop:
        return OccupancyVerdict(OccupancyKind.MATCH, miss, None, changed)

    hyps = [
        h
        for h in hypothesize_moves(expected, seen_b, board)
        if colors_support_move(colors, h.move, board)
    ]
    # Prefer a single unambiguous legal interpretation.
    if len(hyps) == 1:
        return OccupancyVerdict(OccupancyKind.ONE_LEGAL, miss, hyps[0].move, changed)
    # Several promo choices for the same from/to → still one move (UI picks piece).
    if len(hyps) > 1:
        bases = {(h.move.from_square, h.move.to_square) for h in hyps}
        if len(bases) == 1 and all(h.move.promotion for h in hyps):
            src, dst = next(iter(bases))
            return OccupancyVerdict(
                OccupancyKind.ONE_LEGAL,
                miss,
                chess.Move(src, dst),
                changed,
            )
    return OccupancyVerdict(OccupancyKind.ILLEGAL_OR_MULTI, miss, None, changed)


def occupancy_frame_is_noise(
    seen: np.ndarray,
    board: chess.Board,
    verdict: OccupancyVerdict,
) -> bool:
    """Detector garbage, or a one-square flicker. Not a move and not a fault.

    A move changes the piece count by at most one. Thirteen pieces on a full
    board is a failed detect: ignore the frame, and do not enter or leave FAULT
    because of it. One extra box (33 vs 32) is the same — it must not strobe red.
    """
    seen_n = int(np.asarray(seen, dtype=bool).sum())
    expected_n = int(occupancy_from_board(board).sum())
    if abs(seen_n - expected_n) > 2:
        return True
    if verdict.kind in (OccupancyKind.MATCH, OccupancyKind.ONE_LEGAL):
        return False
    if seen_n > expected_n:
        return True
    return verdict.mismatch <= 1


def hypothesize_moves(
    before: np.ndarray,
    after: np.ndarray,
    board: chess.Board,
) -> list[MoveHypothesis]:
    """Match occupancy delta to legal moves."""
    before_b = np.asarray(before, dtype=bool)
    after_b = np.asarray(after, dtype=bool)
    vacated = np.argwhere(before_b & ~after_b)
    filled = np.argwhere(~before_b & after_b)

    candidates: list[MoveHypothesis] = []
    legal = list(board.legal_moves)

    # Standard / promotion: one vacated, one filled
    if len(vacated) == 1 and len(filled) == 1:
        fr, fc = vacated[0]
        tr, tc = filled[0]
        src = _idx_to_square(int(fr), int(fc))
        dst = _idx_to_square(int(tr), int(tc))
        for m in legal:
            if m.from_square == src and m.to_square == dst:
                candidates.append(MoveHypothesis(m, 1.0))

    # Capture: vacated origin, destination stayed occupied
    if len(vacated) == 1 and len(filled) == 0:
        fr, fc = vacated[0]
        src = _idx_to_square(int(fr), int(fc))
        for m in legal:
            if m.from_square == src and board.is_capture(m) and not board.is_en_passant(m):
                dr = 7 - chess.square_rank(m.to_square)
                dc = chess.square_file(m.to_square)
                if after_b[dr, dc]:
                    candidates.append(MoveHypothesis(m, 0.9))

    # En passant: vacated from + captured pawn, filled destination
    if len(vacated) == 2 and len(filled) == 1:
        for m in legal:
            if not board.is_en_passant(m):
                continue
            cap = chess.square(chess.square_file(m.to_square), chess.square_rank(m.from_square))
            vac = {_idx_to_square(int(r), int(c)) for r, c in vacated}
            fill = _idx_to_square(int(filled[0][0]), int(filled[0][1]))
            if vac == {m.from_square, cap} and fill == m.to_square:
                candidates.append(MoveHypothesis(m, 0.92))

    # Castling: king and rook both move — two vacated, two filled
    if len(vacated) == 2 and len(filled) == 2:
        vac = {_idx_to_square(int(r), int(c)) for r, c in vacated}
        fill = {_idx_to_square(int(r), int(c)) for r, c in filled}
        for m in legal:
            if not board.is_castling(m):
                continue
            touched = set(squares_affected_by_move(board, m))
            if vac | fill == touched and vac.isdisjoint(fill):
                candidates.append(MoveHypothesis(m, 0.95))
            elif m.from_square in vac and m.to_square in fill:
                candidates.append(MoveHypothesis(m, 0.9))

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
