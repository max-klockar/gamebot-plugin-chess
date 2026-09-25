from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

import chess
import chess.engine

from chess_plugin.game import skill_to_stockfish


@dataclass
class EngineMove:
    move: chess.Move
    ponder: chess.Move | None = None


class ChessEngine:
    """Stockfish wrapper; falls back to random legal move if binary missing."""

    def __init__(
        self,
        path: str | None = None,
        threads: int = 2,
        hash_mb: int = 64,
    ) -> None:
        self.path = path or shutil.which("stockfish")
        self.threads = threads
        self.hash_mb = hash_mb
        self._engine: chess.engine.SimpleEngine | None = None

    def open(self) -> None:
        if self.path is None:
            return
        try:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"Threads": self.threads, "Hash": self.hash_mb})
        except (OSError, chess.engine.EngineError):
            self._engine = None

    def close(self) -> None:
        if self._engine is not None:
            self._engine.quit()
            self._engine = None

    def __enter__(self) -> ChessEngine:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @property
    def available(self) -> bool:
        return self._engine is not None

    def play(self, board: chess.Board, skill_level: int) -> EngineMove:
        sf_skill, movetime = skill_to_stockfish(skill_level)
        if self._engine is None:
            return EngineMove(self._fallback(board))
        limit = chess.engine.Limit(time=movetime / 1000.0)
        try:
            self._engine.configure({"Skill Level": sf_skill})
        except chess.engine.EngineError:
            pass
        result = self._engine.play(board, limit)
        if result.move is None:
            return EngineMove(self._fallback(board))
        return EngineMove(result.move, result.ponder)

    def _fallback(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise RuntimeError("no legal moves")
        # Prefer captures / checks lightly without Stockfish
        scored = []
        for m in moves:
            score = 0
            if board.is_capture(m):
                score += 2
            board.push(m)
            if board.is_check():
                score += 1
            board.pop()
            scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]


def stockfish_installed() -> bool:
    return shutil.which("stockfish") is not None


def try_version(path: str | None = None) -> str | None:
    bin_path = path or shutil.which("stockfish")
    if not bin_path:
        return None
    try:
        out = subprocess.check_output([bin_path, "--help"], stderr=subprocess.STDOUT, timeout=2)
        return out.decode(errors="ignore").splitlines()[0] if out else bin_path
    except (OSError, subprocess.SubprocessError):
        return bin_path
