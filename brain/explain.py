"""Plain-English reasons for a move, shown in the AI Brain panel."""
from .game import CELL_NAMES, LINE_NAMES, LINES, INFINITE, State, available_moves, other, result


def _line_name(line):
    return LINE_NAMES[LINES.index(line)]


def reason(state: State, cell: int, ai: str, mode: str) -> str:
    """Why the AI's move on `cell` matters, checked in order of importance."""
    human = other(ai)
    played = result(state, cell, mode)
    if played.line:
        return "Completes the " + _line_name(played.line)

    # Would the human have won by playing here?
    theirs = result(State(state.X, state.O, human), cell, mode)
    if theirs.line:
        return "Blocks your " + _line_name(theirs.line)

    # Count the AI's winning threats if it could move again.
    again = State(played.state.X, played.state.O, ai)
    threats = [c for c in available_moves(again) if result(again, c, mode).line]
    if len(threats) >= 2:
        return "Creates a fork: two ways to win"
    if len(threats) == 1:
        return f"Threatens to win on cell {threats[0] + 1}"
    if mode == INFINITE and played.vanished is not None:
        return f"Repositions while cell {played.vanished + 1} vanishes"
    return "Takes the " + CELL_NAMES[cell]
