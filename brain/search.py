"""Adversarial search: plain Minimax and Minimax with Alpha-Beta pruning.

The AI is MAX and the human is MIN; every score is from the AI's side.

Classic Tic-Tac-Toe is small enough to search to the end of the game, so its
scores are exact. In Infinite mode the game can go on forever, so the search
stops at a depth limit and estimates the position there with heuristic()
(a "cutoff test" plus an evaluation function). Iterative deepening runs the
search at depth 1, 2, 3, ... until it proves a result or runs out of time.
"""
import math
import time

from .evaluation import heuristic, utility
from .game import INFINITE, MAX_MARKS, available_moves, is_full, marks, other, position_key, result

INF = math.inf


class SearchTimeout(Exception):
    """The search passed its deadline. The caller keeps the last finished depth."""


class Search:
    """Settings and counters for one search from one root position."""

    def __init__(self, mode, ai, depth_limit, history=None, deadline=None, hints=None):
        self.mode = mode
        self.ai = ai
        self.depth_limit = depth_limit
        # Position key -> how often it has occurred, in the game so far plus
        # along the current search path. Only Infinite mode needs it.
        self.seen = dict(history or {})
        self.deadline = deadline
        # Position key -> best move found by an earlier search. Used only to
        # try that move first, which never changes a result.
        self.hints = hints if hints is not None else {}
        self.nodes = 0   # positions visited
        self.cuts = 0    # times alpha-beta pruned the remaining moves
        self.step = 0    # shared counter for visitOrder / doneOrder in the recorded tree


def expand(state, cell, search, ply):
    """Play `cell` and apply the terminal test to the new position.

    Returns (outcome, terminal_score or None, ended_by_repetition, key), where
    key is the new position's repetition key (Infinite mode only).
    """
    out = result(state, cell, search.mode)
    mover_is_ai = state.to_move == search.ai
    if out.line:
        return out, utility(mover_is_ai, ply + 1), False, None
    if search.mode == INFINITE:
        key = position_key(out.state)
        if search.seen.get(key, 0) >= 2:      # this move makes it the 3rd time
            return out, utility(not mover_is_ai, ply + 1), True, key
        return out, None, False, key
    if is_full(out.state):
        return out, utility(None, ply + 1), False, None
    return out, None, False, None


def _children(state, search, ply):
    """All moves from `state`, expanded and ordered for alpha-beta:
    winning moves first, then the move that was best last time, then the
    static centre-corners-edges order."""
    kids = [(cell,) + expand(state, cell, search, ply) for cell in available_moves(state)]
    hint = search.hints.get(position_key(state))
    maximizing = state.to_move == search.ai

    def rank(kid):
        cell, out, term, _, _ = kid
        if term is not None and (term > 0) == maximizing and term != 0:
            return 0
        return 1 if cell == hint else 2

    kids.sort(key=rank)
    return kids


def alphabeta(state, ply, alpha, beta, search, record=None):
    """Minimax value of `state` with alpha-beta pruning.

    alpha is the best score MAX (the AI) can already guarantee on this path,
    beta the best score MIN (the human) can already guarantee. Once
    alpha >= beta the remaining moves cannot affect the decision higher up,
    so they are skipped. The value returned is the same as plain minimax().

    record: a list that receives one node per move, for the UI's search tree
    (root moves and the replies to them), or None.
    """
    search.nodes += 1
    if search.deadline and search.nodes & 1023 == 0 and time.perf_counter() > search.deadline:
        raise SearchTimeout
    if ply >= search.depth_limit:                      # cutoff test
        return heuristic(state, search.ai, search.mode)

    maximizing = state.to_move == search.ai
    best, best_move = (-INF if maximizing else INF), None
    kids = _children(state, search, ply)
    for i, (cell, out, term, by_repetition, key) in enumerate(kids):
        node = None
        if record is not None:
            search.step += 1
            node = {
                "move": cell, "player": "MAX" if maximizing else "MIN",
                "alpha": alpha, "beta": beta, "value": None,
                "pruned": False, "cutoff": False, "terminal": term is not None,
                "repetition": by_repetition, "vanished": out.vanished,
                "visitOrder": search.step, "doneOrder": 0, "subtree": 0,
                "children": [] if ply == 0 else None,
            }
            record.append(node)
        before = search.nodes

        if term is not None:
            search.nodes += 1                          # the finished position counts as a visit
            value = term
        else:
            if key is not None:
                search.seen[key] = search.seen.get(key, 0) + 1
            if node is not None and ply + 1 >= search.depth_limit:
                node["cutoff"] = True
            value = alphabeta(out.state, ply + 1, alpha, beta, search,
                              node["children"] if node is not None else None)
            if key is not None:
                search.seen[key] -= 1

        if node is not None:
            search.step += 1
            node["value"], node["subtree"], node["doneOrder"] = value, search.nodes - before, search.step

        if maximizing:
            if value > best:
                best, best_move = value, cell
            alpha = max(alpha, best)
        else:
            if value < best:
                best, best_move = value, cell
            beta = min(beta, best)

        if alpha >= beta:                              # prune the remaining moves
            search.cuts += 1
            if record is not None:
                search.step += 1
                for rest in kids[i + 1:]:
                    record.append({
                        "move": rest[0], "player": "MAX" if maximizing else "MIN",
                        "alpha": alpha, "beta": beta, "value": None,
                        "pruned": True, "cutoff": False, "terminal": False,
                        "repetition": False, "vanished": rest[1].vanished,
                        "visitOrder": search.step, "doneOrder": search.step,
                        "subtree": 0, "children": None,
                    })
            break

    search.hints[position_key(state)] = best_move
    return best


def minimax(state, ply, search):
    """Plain Minimax: tries every move and never prunes. Used as the baseline
    for node counts; it returns the same value as alphabeta()."""
    search.nodes += 1
    if ply >= search.depth_limit:
        return heuristic(state, search.ai, search.mode)
    maximizing = state.to_move == search.ai
    values = []
    for cell in available_moves(state):
        out, term, _, key = expand(state, cell, search, ply)
        if term is not None:
            search.nodes += 1
            values.append(term)
            continue
        if key is not None:
            search.seen[key] = search.seen.get(key, 0) + 1
        values.append(minimax(out.state, ply + 1, search))
        if key is not None:
            search.seen[key] -= 1
    return max(values) if maximizing else min(values)


_count_memo = {}


def count_minimax_nodes(state, mode, depth):
    """How many nodes plain minimax() visits from `state` to `depth`.

    Classic: exact. The count only depends on the position, so it is
    memoised instead of actually running the slow search.
    Infinite: an estimate that assumes every branch reaches the depth limit
    (it ignores early wins and repetitions), so it is labelled approximate.
    Returns (count, exact).
    """
    if mode == INFINITE:
        mover, waiting = len(marks(state, state.to_move)), len(marks(state, other(state.to_move)))
        total, width = 1, 1
        for _ in range(depth):
            width *= 9 - mover - waiting
            total += width
            mover = min(MAX_MARKS, mover + 1)
            mover, waiting = waiting, mover
        return total, False

    # In Classic the order the marks were placed doesn't matter, only which cells.
    memo_key = (tuple(sorted(state.X)), tuple(sorted(state.O)), state.to_move, depth)
    if memo_key not in _count_memo:
        n = 1
        if depth > 0:
            for cell in available_moves(state):
                out = result(state, cell, mode)
                if out.line or is_full(out.state) or depth == 1:
                    n += 1
                else:
                    n += count_minimax_nodes(out.state, mode, depth - 1)[0]
        _count_memo[memo_key] = n
    return _count_memo[memo_key], True


def principal_variation(state, first_move, search, max_len):
    """The expected line of play: `first_move`, then the best reply found by
    the search at each step, until the game would end or max_len moves."""
    line, seen = [], dict(search.seen)
    cell, ply = first_move, 0
    while cell is not None and len(line) < max_len and cell in available_moves(state):
        line.append(cell)
        out = result(state, cell, search.mode)
        if out.line or (search.mode != INFINITE and is_full(out.state)):
            break
        if search.mode == INFINITE:
            key = position_key(out.state)
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                break
        state, ply = out.state, ply + 1
        cell = search.hints.get(position_key(state))
    return line
