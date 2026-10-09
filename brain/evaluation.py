"""Utility and evaluation functions.

Inside the search every score is from the AI's point of view:

* terminal positions score +-(WIN - plies). A win found sooner scores higher
  and a loss found later scores higher, so the agent takes the fastest win
  and, when lost, the slowest loss. A draw scores 0.
* positions cut off at the depth limit get a heuristic estimate in
  [-HEURISTIC_MAX, +HEURISTIC_MAX], always far from any terminal score.

The UI uses a smaller -10..+10 scale; to_display() converts between them.
"""
import math

from .game import CLASSIC, INFINITE, LINES, MAX_MARKS, marks, other

WIN = 1000
HEURISTIC_MAX = 30
_EXACT_FLOOR = WIN - 100  # |score| at or above this means a proven result


def utility(winner_is_ai, plies: int) -> int:
    """Depth-adjusted utility of a finished game, `plies` moves from the root.
    winner_is_ai is True, False, or None for a draw."""
    if winner_is_ai is None:
        return 0
    return WIN - plies if winner_is_ai else -(WIN - plies)


def is_proven(score: float) -> bool:
    """True if the score is a forced win or loss rather than an estimate."""
    return abs(score) >= _EXACT_FLOOR


def plies_to_end(score: float) -> int:
    """For a proven score, the number of moves until the game ends."""
    return int(WIN - abs(score))


def _live(cells, mode):
    """Marks that can still be part of a line on the owner's next move.
    In Infinite mode a player with 3 marks loses the oldest when they move,
    so that mark can't help complete a line."""
    if mode == INFINITE and len(cells) == MAX_MARKS:
        return cells[1:]
    return cells


def heuristic(state, ai: str, mode: str) -> int:
    """Estimate a non-terminal position from the AI's side.

    For every line still open to one player only, that player scores the
    square of their live marks on it (1 or 4). A threat (two live marks and an
    empty third cell) for the player about to move is almost a win, so it
    scores close to the maximum.
    """
    human = other(ai)
    ai_all, hu_all = marks(state, ai), marks(state, human)
    ai_live, hu_live = _live(ai_all, mode), _live(hu_all, mode)
    occupied = ai_all + hu_all
    score = 0
    threats = {ai: 0, human: 0}
    for line in LINES:
        a_all = sum(c in ai_all for c in line)
        h_all = sum(c in hu_all for c in line)
        a_live = sum(c in ai_live for c in line)
        h_live = sum(c in hu_live for c in line)
        if h_all == 0:
            score += a_live * a_live
            if a_live == 2 and any(c not in occupied for c in line):
                threats[ai] += 1
        if a_all == 0:
            score -= h_live * h_live
            if h_live == 2 and any(c not in occupied for c in line):
                threats[human] += 1
    mover = state.to_move
    if threats[mover]:
        return HEURISTIC_MAX - 5 if mover == ai else -(HEURISTIC_MAX - 5)
    score += 6 * (threats[ai] - threats[human])
    return max(-HEURISTIC_MAX + 6, min(HEURISTIC_MAX - 6, score))


def to_display(score, mode: str):
    """Convert an internal score to the UI's -10..+10 scale.

    Classic: win in p moves = 10 - p, loss = -(10 - p), draw = 0.
    Infinite: a proven result shows as +-(10 - ceil(p/2)), never below 5;
    estimates show between -3 and +3. The UI relies on |v| >= 5 meaning
    proven in Infinite mode.
    """
    if score is None:
        return None
    if score in (math.inf, -math.inf):
        return score
    sign = 1 if score > 0 else -1
    if is_proven(score):
        p = plies_to_end(score)
        if mode == CLASSIC:
            return sign * max(1, 10 - p)
        return sign * max(5, 10 - math.ceil(p / 2))
    if score == 0:
        return 0
    return max(-3, min(3, round(score / 10))) or sign
