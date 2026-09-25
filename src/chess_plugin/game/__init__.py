from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from io import StringIO

import chess
import chess.pgn

from chess_plugin.db import Database, GameRecord, Profile


class SkillBand(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


# Bands map to coarse ranges; finer skill_level 1-10 within band.
BAND_LEVEL_RANGES: dict[SkillBand, range] = {
    SkillBand.BEGINNER: range(1, 4),
    SkillBand.INTERMEDIATE: range(4, 7),
    SkillBand.ADVANCED: range(7, 9),
    SkillBand.EXPERT: range(9, 11),
}


def skill_to_stockfish(skill_level: int) -> tuple[int, int]:
    """Map 1-10 coaching level → (Stockfish Skill Level 0-20, movetime ms)."""
    skill_level = max(1, min(10, skill_level))
    sf_skill = int(round((skill_level - 1) * 20 / 9))
    movetime = 100 + skill_level * 80
    return sf_skill, movetime


@dataclass
class CheckHighlight:
    from_sq: chess.Square
    to_sq: chess.Square


@dataclass
class RevertHint:
    from_sq: chess.Square
    to_sq: chess.Square
    reason: str


@dataclass
class GameEvent:
    kind: str
    payload: dict = field(default_factory=dict)


class Phase(Enum):
    SPLASH = auto()
    CALIBRATE = auto()
    PROFILE = auto()
    SKILL_BAND = auto()
    SKILL_LEVEL = auto()
    PLAY = auto()
    PROMOTE = auto()
    GAME_OVER = auto()


class ChessSession:
    """Rules + coaching events; vision/UI sit outside."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.phase = Phase.SPLASH
        self.board = chess.Board()
        self.profile: Profile | None = None
        self.skill_band = SkillBand.INTERMEDIATE
        self.skill_level = 5
        self.game: GameRecord | None = None
        self.pending_promotion: chess.Move | None = None
        self.last_engine_move: chess.Move | None = None
        self.last_engine_promotion: chess.PieceType | None = None
        self.revert_hint: RevertHint | None = None
        self.human_is_white = True
        self.events: list[GameEvent] = []

    def emit(self, kind: str, **payload: object) -> None:
        self.events.append(GameEvent(kind, dict(payload)))

    def drain_events(self) -> list[GameEvent]:
        out = self.events
        self.events = []
        return out

    def start_calibration(self) -> None:
        self.phase = Phase.CALIBRATE

    def calibration_done(self) -> None:
        self.phase = Phase.PROFILE

    def select_profile(self, profile: Profile) -> None:
        self.profile = profile
        self.skill_band = SkillBand(profile.skill_band)
        self.skill_level = profile.skill_level
        self.phase = Phase.SKILL_BAND

    def select_skill_band(self, band: SkillBand) -> None:
        self.skill_band = band
        levels = list(BAND_LEVEL_RANGES[band])
        self.skill_level = levels[len(levels) // 2]
        self.phase = Phase.SKILL_LEVEL

    def select_skill_level(self, level: int) -> None:
        allowed = list(BAND_LEVEL_RANGES[self.skill_band])
        if level not in allowed:
            raise ValueError(f"level {level} not in {allowed}")
        self.skill_level = level
        if self.profile is not None:
            self.db.update_profile_skill(self.profile.id, self.skill_band.value, level)
        self._start_game()

    def _start_game(self) -> None:
        self.board = chess.Board()
        self.pending_promotion = None
        self.last_engine_move = None
        self.last_engine_promotion = None
        self.revert_hint = None
        pid = self.profile.id if self.profile else None
        self.game = self.db.start_game(pid, self.skill_level)
        self.phase = Phase.PLAY
        self.emit("game_started", game_id=self.game.id)

    def check_highlights(self) -> list[CheckHighlight]:
        if not self.board.is_check():
            return []
        king_sq = self.board.king(self.board.turn)
        if king_sq is None:
            return []
        attackers = self.board.attackers(not self.board.turn, king_sq)
        return [CheckHighlight(a, king_sq) for a in attackers]

    def try_human_move(self, move: chess.Move) -> bool:
        """Apply a detected human move. Returns True if accepted."""
        if self.phase == Phase.PROMOTE and self.pending_promotion is not None:
            return False
        if self.phase != Phase.PLAY:
            return False
        if self.board.turn != (chess.WHITE if self.human_is_white else chess.BLACK):
            return False

        # Incomplete promotion from occupancy: ask UI
        if self._needs_promotion_choice(move):
            promo_move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            if promo_move not in self.board.legal_moves and move not in self.board.legal_moves:
                # try without promo flag
                base = chess.Move(move.from_square, move.to_square)
                legal_promos = [
                    m for m in self.board.legal_moves
                    if m.from_square == base.from_square and m.to_square == base.to_square and m.promotion
                ]
                if not legal_promos:
                    self._reject_move(move, "illegal")
                    return False
            self.pending_promotion = chess.Move(move.from_square, move.to_square)
            self.phase = Phase.PROMOTE
            self.emit("promotion_needed", from_sq=move.from_square, to_sq=move.to_square)
            return True

        if move not in self.board.legal_moves:
            # Maybe user forgot promotion piece — legal queens/etc.
            candidates = [
                m for m in self.board.legal_moves
                if m.from_square == move.from_square and m.to_square == move.to_square
            ]
            if len(candidates) == 1:
                move = candidates[0]
            elif candidates and all(c.promotion for c in candidates):
                self.pending_promotion = chess.Move(move.from_square, move.to_square)
                self.phase = Phase.PROMOTE
                self.emit("promotion_needed", from_sq=move.from_square, to_sq=move.to_square)
                return True
            else:
                self._reject_move(move, "illegal")
                return False

        self._apply_and_record(move, by_engine=False)
        self.revert_hint = None
        return True

    def _needs_promotion_choice(self, move: chess.Move) -> bool:
        piece = self.board.piece_at(move.from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        to_rank = chess.square_rank(move.to_square)
        return (piece.color == chess.WHITE and to_rank == 7) or (
            piece.color == chess.BLACK and to_rank == 0
        )

    def confirm_promotion(self, piece_type: chess.PieceType) -> bool:
        if self.phase != Phase.PROMOTE or self.pending_promotion is None:
            return False
        move = chess.Move(
            self.pending_promotion.from_square,
            self.pending_promotion.to_square,
            promotion=piece_type,
        )
        if move not in self.board.legal_moves:
            self._reject_move(move, "illegal_promotion")
            self.pending_promotion = None
            self.phase = Phase.PLAY
            return False
        self.pending_promotion = None
        self.phase = Phase.PLAY
        self._apply_and_record(move, by_engine=False)
        return True

    def apply_engine_move(self, move: chess.Move) -> bool:
        if self.phase != Phase.PLAY:
            return False
        if move not in self.board.legal_moves:
            return False
        promo = move.promotion
        self._apply_and_record(move, by_engine=True)
        self.last_engine_move = move
        self.last_engine_promotion = promo
        self.emit("engine_move", uci=move.uci(), promotion=promo)
        return True

    def _reject_move(self, move: chess.Move, reason: str) -> None:
        self.revert_hint = RevertHint(move.to_square, move.from_square, reason)
        self.emit("illegal_move", uci=move.uci(), reason=reason)

    def _apply_and_record(self, move: chess.Move, by_engine: bool) -> None:
        san = self.board.san(move)
        self.board.push(move)
        if self.game is not None:
            ply = len(self.board.move_stack)
            self.db.add_move(self.game.id, ply, move.uci(), san, by_engine)
            self.db.finish_game(self.game.id, self._result_or_ongoing(), self.pgn())
        self.emit("move_played", uci=move.uci(), san=san, by_engine=by_engine)
        if self.board.is_game_over():
            self.phase = Phase.GAME_OVER
            result = self.board.result()
            if self.game is not None:
                self.db.finish_game(self.game.id, result, self.pgn())
            self.emit("game_over", result=result)

    def _result_or_ongoing(self) -> str:
        if self.board.is_game_over():
            return self.board.result()
        return "*"

    def pgn(self) -> str:
        game = chess.pgn.Game()
        if self.profile is not None:
            game.headers["White"] = self.profile.name if self.human_is_white else "Schack"
            game.headers["Black"] = "Schack" if self.human_is_white else self.profile.name
        else:
            game.headers["White"] = "Guest" if self.human_is_white else "Schack"
            game.headers["Black"] = "Schack" if self.human_is_white else "Guest"
        game.headers["Result"] = self.board.result(claim_draw=True) if self.board.is_game_over() else "*"
        node = game
        replay = chess.Board()
        for mv in self.board.move_stack:
            node = node.add_variation(mv)
            replay.push(mv)
        buf = StringIO()
        print(game, file=buf, end="")
        return buf.getvalue()

    def is_human_turn(self) -> bool:
        human_color = chess.WHITE if self.human_is_white else chess.BLACK
        return self.phase == Phase.PLAY and self.board.turn == human_color
