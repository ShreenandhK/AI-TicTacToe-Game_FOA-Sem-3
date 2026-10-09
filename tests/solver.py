"""Independent exact solver for Infinite mode, used only to check the AI.

Retrograde analysis over every reachable position (about 116,000): positions
where the mover can win at once are wins; a position is lost when every move
leads to a position the opponent wins; the rest can loop forever. It ignores
the repetition rule, which cannot trigger within the AI's search horizon when
the game history is empty.
"""
from collections import deque

from brain.game import INFINITE, State, available_moves, result


def solve():
    """Return {state: (value, plies)} for the player to move: +1 win, -1 loss.
    Positions missing from the dict are unresolved (endless with best play)."""
    start = [State((), (), "X"), State((), (), "O")]
    succ, pred, seen = {}, {}, set(start)
    queue = deque(start)
    while queue:
        s = queue.popleft()
        kids = []
        for cell in available_moves(s):
            out = result(s, cell, INFINITE)
            kids.append((bool(out.line), out.state))
            if not out.line:
                pred.setdefault(out.state, []).append(s)
                if out.state not in seen:
                    seen.add(out.state)
                    queue.append(out.state)
        succ[s] = kids

    value, remaining, work = {}, {}, deque()
    for s, kids in succ.items():
        if any(won for won, _ in kids):
            value[s] = (1, 1)
            work.append(s)
        else:
            remaining[s] = len(kids)
    while work:
        s = work.popleft()
        v, d = value[s]
        for p in pred.get(s, []):
            if p in value:
                continue
            if v == -1:
                value[p] = (1, d + 1)
                work.append(p)
            else:
                remaining[p] -= 1
                if remaining[p] == 0:
                    value[p] = (-1, max(value[k][1] for _, k in succ[p]) + 1)
                    work.append(p)
    return value
