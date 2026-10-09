"""The agent: chooses a move for each difficulty and explains it to the UI.

analyze(game, opts) is the only function the UI calls. It returns the object
the AI Brain panel renders; README.md lists every field.

Difficulty levels
* Unbeatable - full Minimax with Alpha-Beta pruning. Classic searches the
  whole game tree, so it never loses. Infinite deepens the search one move at
  a time up to INFINITE_MAX_DEPTH, enough to find the 13-move forced win from
  the empty board.
* Normal - simulates a human. It searches only NORMAL_DEPTH moves ahead (it
  sees immediate wins and threats but not forks), and with probability
  slipChance it plays a move it knows is worse. It never misses a win in one.
"""
import random
import time
from typing import NamedTuple

from .evaluation import is_proven, plies_to_end, to_display
from .explain import reason
from .game import (CLASSIC, INFINITE, MAX_MARKS, State, available_moves, marks,
                   position_key, result, validate)
from .search import (INF, Search, SearchTimeout, alphabeta, count_minimax_nodes,
                     expand, minimax, principal_variation)

UNBEATABLE, NORMAL = "unbeatable", "normal"
CLASSIC_DEPTH = 9          # the whole game tree
INFINITE_MAX_DEPTH = 14    # the forced win from the empty board is 13 moves deep
NORMAL_DEPTH = 2
SLIP_CHANCE = 0.3

# Time budget in seconds
SOFT_LIMIT = 0.8           # don't start another depth after this
HARD_LIMIT = 2.5           # abandon an unfinished depth after this
SCORING_LIMIT = 1.5        # scoring every candidate move
MEASURE_MINIMAX_UP_TO = 60_000  # run real plain Minimax for timing when it is this small

ENGINE = "python"
ENGINE_VERSION = "1.0"


class Iteration(NamedTuple):
    value: float     # root value, internal scale
    move: int        # best root move
    search: Search
    tree: list       # recorded nodes for the UI
    depth: int
    ms: float


def _deepen(state, mode, ai, max_depth, history, hints, iterative):
    """Run alpha-beta at depth 1, 2, ... max_depth (or only max_depth when
    not iterative). Returns (last finished Iteration, deepening log)."""
    start = time.perf_counter()
    last, log = None, []
    for depth in range(1 if iterative else max_depth, max_depth + 1):
        search = Search(mode, ai, depth, history,
                        deadline=start + HARD_LIMIT if last else None, hints=hints)
        tree, t = [], time.perf_counter()
        try:
            value = alphabeta(state, 0, -INF, INF, search, tree)
        except SearchTimeout:
            break
        ms = (time.perf_counter() - t) * 1000
        last = Iteration(value, hints[position_key(state)], search, tree, depth, ms)
        log.append({"depth": depth, "nodes": search.nodes, "ms": round(ms, 1)})
        if is_proven(value):          # forced win or loss: a deeper search can't change it
            break
        if time.perf_counter() - start > SOFT_LIMIT:
            break
    return last, log


def _score_candidates(state, mode, ai, depth, history, hints, fallback_tree):
    """Exact (full-window) alpha-beta score of every legal move, for the
    move-score board. Falls back to the decision tree's value on timeout."""
    deadline = time.perf_counter() + SCORING_LIMIT
    mine = marks(state, ai)
    vanishes = mine[0] if mode == INFINITE and len(mine) == MAX_MARKS else None
    scores, nodes = [], 0
    for cell in available_moves(state):
        search = Search(mode, ai, depth, history, deadline=deadline, hints=hints)
        out, term, _, key = expand(state, cell, search, 0)
        if term is not None:
            score, exact = term, True
        else:
            if key is not None:
                search.seen[key] = search.seen.get(key, 0) + 1
            try:
                score = alphabeta(out.state, 1, -INF, INF, search)
                exact = mode == CLASSIC or is_proven(score)
            except SearchTimeout:
                node = next((n for n in fallback_tree if n["move"] == cell and not n["pruned"]), None)
                score, exact = (node["value"] if node else 0), False
        nodes += search.nodes
        scores.append({"cell": cell, "score": score, "exact": exact, "vanishes": vanishes,
                       "winsNow": bool(out.line)})
    return scores, nodes


def _plain_minimax(state, mode, ai, depth, history, ab_nodes, ab_ms):
    """Node count and time for plain Minimax at the same depth. Runs the real
    search when it is small enough, otherwise counts or estimates."""
    count, exact = count_minimax_nodes(state, mode, depth)
    if count <= MEASURE_MINIMAX_UP_TO:
        search = Search(mode, ai, depth, history)
        t = time.perf_counter()
        minimax(state, 0, search)
        return search.nodes, True, (time.perf_counter() - t) * 1000, True
    count = max(count, ab_nodes + 1)
    return count, exact, ab_ms * count / max(1, ab_nodes), False


def _display_tree(nodes, mode):
    for n in nodes:
        n["value"] = to_display(n["value"], mode)
        n["alpha"] = to_display(n["alpha"], mode)
        n["beta"] = to_display(n["beta"], mode)
        if n.get("children"):
            _display_tree(n["children"], mode)
    return nodes


def best_move(state: State, mode: str = CLASSIC, history=None, max_depth=None) -> int:
    """Root decision for the player to move: the move with the best
    alpha-beta value (Unbeatable play)."""
    depth = CLASSIC_DEPTH if mode == CLASSIC else (max_depth or INFINITE_MAX_DEPTH)
    last, _ = _deepen(state, mode, state.to_move, depth, history or {}, {}, mode == INFINITE)
    return last.move


def analyze(game: dict, opts: dict = None) -> dict:
    """Choose the AI's move and describe the reasoning for the UI.

    game: {mode, board: {X: [...], O: [...]} oldest first, toMove, history: {positionKey: count}}
    opts: {ai, difficulty: 'unbeatable'|'normal', maxDepth, slipChance, seed, forceMove}
    """
    opts = opts or {}
    mode = game.get("mode")
    board = game.get("board") or {}
    state = State(tuple(board.get("X") or ()), tuple(board.get("O") or ()), game.get("toMove"))
    validate(state, mode)
    ai = opts.get("ai") or state.to_move
    if ai != state.to_move:
        raise ValueError("it is not the AI's turn")
    difficulty = opts.get("difficulty") or UNBEATABLE
    if difficulty not in (UNBEATABLE, NORMAL):
        raise ValueError(f"unknown difficulty {difficulty!r}")
    moves = available_moves(state)
    if not moves:
        raise ValueError("there are no legal moves")
    history = {str(k): int(v) for k, v in (game.get("history") or {}).items()}

    full_depth = CLASSIC_DEPTH if mode == CLASSIC else max(1, min(16, int(opts.get("maxDepth") or INFINITE_MAX_DEPTH)))
    depth_limit = NORMAL_DEPTH if difficulty == NORMAL else full_depth
    hints = {}

    # 1. The decision search, recorded for the search-tree card.
    decision, deepening = _deepen(state, mode, ai, depth_limit, history, hints, mode == INFINITE)
    time_ms = sum(d["ms"] for d in deepening)

    # 2. A full-strength search to know the true value of every move.
    #    Unbeatable already has it; Normal needs it to show "Optimal vs Played".
    if difficulty == UNBEATABLE:
        reference = decision
    else:
        reference, _ = _deepen(state, mode, ai, full_depth, history, hints, mode == INFINITE)
    candidates, scoring_nodes = _score_candidates(state, mode, ai, reference.depth, history, hints, reference.tree)
    by_cell = {c["cell"]: c for c in candidates}
    best_score = max(c["score"] for c in candidates)

    # 3. Pick the move.
    slip_kind = None
    if opts.get("forceMove") is not None:
        chosen = int(opts["forceMove"])
        if chosen not in by_cell:
            raise ValueError("forceMove is not a legal move")
    elif difficulty == UNBEATABLE:
        chosen = decision.move
        if by_cell[chosen]["score"] < best_score:      # only if scoring searched deeper
            chosen = next(c["cell"] for c in candidates if c["score"] == best_score)
    else:
        rng = random.Random(opts.get("seed"))
        slip_chance = float(opts.get("slipChance", SLIP_CHANCE))
        chosen = decision.move
        win_now = next((c["cell"] for c in candidates if c["winsNow"]), None)
        if win_now is not None:
            chosen = win_now
        elif rng.random() < slip_chance:
            worse = sorted((c for c in candidates if c["score"] < best_score), key=lambda c: -c["score"])
            if worse:
                chosen = (worse[0] if rng.random() < 0.7 else rng.choice(worse))["cell"]
                slip_kind = "random"
    pick = by_cell[chosen]
    optimal = pick if pick["score"] == best_score else next(c for c in candidates if c["score"] == best_score)
    slip = pick["score"] < best_score
    if slip and slip_kind is None:
        slip_kind = "shallow"      # the 2-move search missed something

    score = pick["score"]
    proven = is_proven(score)
    exact = pick["exact"]
    ab_nodes = decision.search.nodes
    mm_nodes, mm_exact, mm_ms, mm_measured = _plain_minimax(
        state, mode, ai, decision.depth, history, ab_nodes, decision.ms)
    pv_search = Search(mode, ai, reference.depth, history, hints=hints)
    root_steps = decision.search.step + 1

    return {
        "engine": ENGINE, "engineVersion": ENGINE_VERSION,
        "mode": mode, "board": {"X": list(state.X), "O": list(state.O)}, "toMove": state.to_move, "ai": ai,
        "difficulty": difficulty,
        "vanishesNext": ({"X": state.X[0] if len(state.X) == MAX_MARKS else None,
                          "O": state.O[0] if len(state.O) == MAX_MARKS else None}
                         if mode == INFINITE else None),
        "chosenMove": chosen, "optimalMove": optimal["cell"],
        "optimalValue": to_display(best_score, mode), "value": to_display(score, mode),
        "slip": slip, "slipKind": slip_kind,
        "reason": reason(state, chosen, ai, mode),
        "exact": exact,
        "forcedWinIn": plies_to_end(score) if exact and proven and score > 0 else None,
        "forcedLossIn": plies_to_end(score) if exact and proven and score < 0 else None,
        "candidates": [{"cell": c["cell"], "score": to_display(c["score"], mode),
                        "exact": c["exact"], "vanishes": c["vanishes"]} for c in candidates],
        "nodes": {"minimax": mm_nodes, "alphaBeta": ab_nodes},
        "minimaxExact": mm_exact, "minimaxMeasured": mm_measured,
        "nodesTotal": sum(d["nodes"] for d in deepening) + scoring_nodes,
        "depthLimit": depth_limit, "depthReached": decision.depth, "scoreDepth": reference.depth,
        "timeMs": round(max(0.1, time_ms), 1), "minimaxTimeMs": round(max(0.1, mm_ms), 1),
        "principalVariation": principal_variation(state, chosen, pv_search, min(reference.depth, 9)),
        "deepening": deepening,
        "repetition": {"count": history.get(position_key(state), 1),
                       "losingCells": [c for c in moves if mode == INFINITE
                                       and not result(state, c, mode).line
                                       and history.get(position_key(result(state, c, mode).state), 0) >= 2]},
        "tree": {"move": None, "player": "MAX", "value": to_display(decision.value, mode),
                 "alpha": -INF, "beta": INF, "pruned": False, "cutoff": False, "terminal": False,
                 "visitOrder": 0, "doneOrder": root_steps,
                 "children": _display_tree(decision.tree, mode)},
        "maxStep": root_steps,
        "cuts": decision.search.cuts,
    }
