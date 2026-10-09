"""Game model: states, actions, transition model and terminal test.

Two modes share one 3x3 board:

* classic  - standard Tic-Tac-Toe. A full board with no line is a draw.
* infinite - each player keeps at most 3 marks. Placing a 4th removes that
             player's oldest mark, so the board never fills. A position that
             occurs for the 3rd time loses for the player who caused it, so
             every game ends with a winner.

A state is (X, O, to_move). X and O are tuples of cells, oldest first, so the
age of every mark is part of the state. Cells are 0-8 in reading order; the UI
shows them as 1-9.
"""
from typing import NamedTuple, Optional, Tuple

CLASSIC = "classic"
INFINITE = "infinite"
MODES = (CLASSIC, INFINITE)
MAX_MARKS = 3  # marks per player in Infinite mode

LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8),
         (0, 3, 6), (1, 4, 7), (2, 5, 8),
         (0, 4, 8), (2, 4, 6))
LINE_NAMES = ("top row 1-2-3", "middle row 4-5-6", "bottom row 7-8-9",
              "left column 1-4-7", "middle column 2-5-8", "right column 3-6-9",
              "diagonal 1-5-9", "diagonal 3-5-7")
CELL_NAMES = ("top-left corner", "top edge", "top-right corner",
              "left edge", "centre", "right edge",
              "bottom-left corner", "bottom edge", "bottom-right corner")

# Centre, then corners, then edges. Trying strong moves first lets
# alpha-beta cut more branches; it never changes the result.
MOVE_ORDER = (4, 0, 2, 6, 8, 1, 3, 5, 7)

_LINE_MASKS = tuple((1 << a) | (1 << b) | (1 << c) for a, b, c in LINES)


class State(NamedTuple):
    X: Tuple[int, ...]
    O: Tuple[int, ...]
    to_move: str


class Outcome(NamedTuple):
    """What happens when a move is played."""
    state: State               # the position after the move
    vanished: Optional[int]    # Infinite mode: the mover's mark that disappeared
    line: Optional[tuple]      # the line the mover completed, if any


def other(player: str) -> str:
    return "O" if player == "X" else "X"


def initial_state(first: str = "X") -> State:
    return State((), (), first)


def marks(state: State, player: str) -> Tuple[int, ...]:
    return state.X if player == "X" else state.O


def winning_line(cells) -> Optional[tuple]:
    """The line formed by these cells, or None."""
    m = 0
    for c in cells:
        m |= 1 << c
    for line, lm in zip(LINES, _LINE_MASKS):
        if m & lm == lm:
            return line
    return None


def available_moves(state: State) -> list:
    """Successor function: the empty cells, in search order."""
    taken = state.X + state.O
    return [c for c in MOVE_ORDER if c not in taken]


def result(state: State, cell: int, mode: str) -> Outcome:
    """Transition model: place the mover's mark on `cell`.

    In Infinite mode the mover's oldest mark vanishes once they would have 4,
    and only the 3 remaining marks count towards a line.
    """
    mover = state.to_move
    mine = marks(state, mover) + (cell,)
    vanished = None
    if mode == INFINITE and len(mine) > MAX_MARKS:
        vanished, mine = mine[0], mine[1:]
    after = State(mine, state.O, "O") if mover == "X" else State(state.X, mine, "X")
    return Outcome(after, vanished, winning_line(mine))


def is_full(state: State) -> bool:
    return len(state.X) + len(state.O) == 9


def position_key(state: State) -> str:
    """Key used to count repetitions. Same format as the UI's keyOf(), so the
    game history the UI sends can be read directly. Mark order is included,
    because the ages of the marks are part of the position."""
    return "".join(map(str, state.X)) + "|" + "".join(map(str, state.O)) + "|" + state.to_move


def winner(state: State) -> Optional[str]:
    """Goal test: 'X' or 'O' if that player has three in a row."""
    if winning_line(state.X):
        return "X"
    if winning_line(state.O):
        return "O"
    return None


def validate(state: State, mode: str) -> None:
    """Raise ValueError if `state` cannot occur in a game of `mode`."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    if state.to_move not in ("X", "O"):
        raise ValueError("to_move must be 'X' or 'O'")
    cells = state.X + state.O
    if any(not isinstance(c, int) or not 0 <= c <= 8 for c in cells):
        raise ValueError("cells must be integers 0-8")
    if len(set(cells)) != len(cells):
        raise ValueError("a cell is used twice")
    if mode == INFINITE and (len(state.X) > MAX_MARKS or len(state.O) > MAX_MARKS):
        raise ValueError("Infinite mode allows at most 3 marks per player")
    if abs(len(state.X) - len(state.O)) > 1:
        raise ValueError("mark counts are out of balance")
    if winner(state):
        raise ValueError("the game is already won")
