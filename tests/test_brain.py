"""Tests for the brain. Run from the AI-TicTacToe folder:

    py -m unittest discover -s tests -v

Set QUICK=1 to skip the exhaustive Infinite-mode test (every human strategy).
"""
import json
import os
import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain import analyze, best_move  # noqa: E402
from brain.evaluation import is_proven  # noqa: E402
from brain.game import (CLASSIC, INFINITE, State, available_moves, is_full,  # noqa: E402
                        position_key, result, validate)
from brain.search import INF, Search, alphabeta, count_minimax_nodes, minimax  # noqa: E402
from server import jsonable  # noqa: E402
from solver import solve  # noqa: E402

EDGES = {1, 3, 5, 7}


def play(mode, cells, first="X"):
    s = State((), (), first)
    for c in cells:
        s = result(s, c, mode).state
    return s


def reachable(mode, max_plies):
    """Every non-terminal position reachable within max_plies, X moving first."""
    frontier, found = [State((), (), "X")], {State((), (), "X")}
    for _ in range(max_plies):
        nxt = []
        for s in frontier:
            for c in available_moves(s):
                out = result(s, c, mode)
                if out.line or (mode == CLASSIC and is_full(out.state)) or out.state in found:
                    continue
                found.add(out.state)
                nxt.append(out.state)
        frontier = nxt
    return list(found)


_SOLVED = None


def solved():
    global _SOLVED
    if _SOLVED is None:
        _SOLVED = solve()
    return _SOLVED


class RulesTest(unittest.TestCase):
    def test_oldest_mark_vanishes_in_infinite(self):
        s = play(INFINITE, [0, 4, 1, 8, 6, 5])          # X: 0 1 6, O: 4 8 5
        out = result(s, 2, INFINITE)                    # X's 4th mark
        self.assertEqual(out.vanished, 0)
        self.assertEqual(out.state.X, (1, 6, 2))
        self.assertIsNone(out.line)                     # 0-1-2 is broken because 0 vanished

    def test_classic_keeps_every_mark(self):
        s = play(CLASSIC, [0, 4, 1, 8])
        out = result(s, 2, CLASSIC)
        self.assertIsNone(out.vanished)
        self.assertEqual(out.line, (0, 1, 2))

    def test_position_key_matches_ui(self):
        # UI: p.X.join('') + '|' + p.O.join('') + '|' + p.toMove
        self.assertEqual(position_key(State((4, 0), (8,), "O")), "40|8|O")

    def test_validate_rejects_bad_positions(self):
        for bad in [State((0, 0), (), "O"), State((0, 1, 2), (3, 4), "O"), State((9,), (), "O")]:
            with self.assertRaises(ValueError):
                validate(bad, CLASSIC)
        with self.assertRaises(ValueError):
            validate(State((0, 1, 5, 7), (2, 3, 4), "O"), INFINITE)


class SearchTest(unittest.TestCase):
    def test_alphabeta_equals_minimax_classic(self):
        for s in reachable(CLASSIC, 4):
            a = alphabeta(s, 0, -INF, INF, Search(CLASSIC, s.to_move, 9))
            m = minimax(s, 0, Search(CLASSIC, s.to_move, 9))
            self.assertEqual(a, m, s)

    def test_alphabeta_equals_minimax_infinite(self):
        rng = random.Random(7)
        for s in rng.sample(reachable(INFINITE, 8), 150):
            for depth in (3, 5):
                a = alphabeta(s, 0, -INF, INF, Search(INFINITE, s.to_move, depth))
                m = minimax(s, 0, Search(INFINITE, s.to_move, depth))
                self.assertEqual(a, m, (s, depth))

    def test_alphabeta_visits_fewer_nodes(self):
        s = State((), (), "X")
        ab, mm = Search(CLASSIC, "X", 9), Search(CLASSIC, "X", 9)
        alphabeta(s, 0, -INF, INF, ab)
        minimax(s, 0, mm)
        self.assertEqual(mm.nodes, 549_946)             # the full Tic-Tac-Toe game tree
        self.assertLess(ab.nodes, mm.nodes // 20)

    def test_minimax_count_is_exact_in_classic(self):
        for cells in ([], [4], [0, 4], [4, 0, 8]):
            s = play(CLASSIC, cells)
            search = Search(CLASSIC, s.to_move, 9)
            minimax(s, 0, search)
            self.assertEqual(count_minimax_nodes(s, CLASSIC, 9), (search.nodes, True))


class ClassicAgentTest(unittest.TestCase):
    def outcomes(self, ai):
        """Play the Unbeatable AI against every possible human strategy."""
        cache, results = {}, []

        def explore(s):
            if s.to_move == ai:
                if s not in cache:
                    cache[s] = best_move(s, CLASSIC)
                moves = [cache[s]]
            else:
                moves = available_moves(s)
            for c in moves:
                out = result(s, c, CLASSIC)
                if out.line:
                    results.append(s.to_move)
                elif is_full(out.state):
                    results.append("draw")
                else:
                    explore(out.state)

        explore(State((), (), "X"))
        return results

    def test_never_loses_moving_first(self):
        self.assertNotIn("O", self.outcomes("X"))

    def test_never_loses_moving_second(self):
        self.assertNotIn("X", self.outcomes("O"))

    def test_takes_fastest_win(self):
        s = play(CLASSIC, [0, 3, 1, 4])                 # X can win now on 2
        a = analyze({"mode": CLASSIC, "board": {"X": list(s.X), "O": list(s.O)}, "toMove": "X"}, {"ai": "X"})
        self.assertEqual(a["chosenMove"], 2)
        self.assertEqual(a["forcedWinIn"], 1)
        self.assertEqual(a["value"], 9)


class InfiniteAgentTest(unittest.TestCase):
    def test_finds_the_13_move_forced_win(self):
        a = analyze({"mode": INFINITE, "board": {"X": [], "O": []}, "toMove": "X"}, {"ai": "X"})
        self.assertIn(a["chosenMove"], EDGES)
        self.assertTrue(a["exact"])
        self.assertEqual(a["forcedWinIn"], 13)

    def test_agrees_with_exact_solver(self):
        """From positions the solver says are won within 13 moves the AI must
        keep the win; from lost positions it must lose as slowly as possible."""
        table, rng = solved(), random.Random(3)
        won = [s for s, (v, d) in table.items() if v == 1 and d <= 13]
        lost = [s for s, (v, d) in table.items() if v == -1 and d <= 12]
        for s in rng.sample(won, 200):
            out = result(s, best_move(s, INFINITE), INFINITE)
            self.assertTrue(out.line or table.get(out.state, (0,))[0] == -1, s)
        for s in rng.sample(lost, 100):
            longest = max(table[result(s, c, INFINITE).state][1] for c in available_moves(s))
            got = table[result(s, best_move(s, INFINITE), INFINITE).state][1]
            self.assertEqual(got, longest, s)

    @unittest.skipIf(os.environ.get("QUICK"), "QUICK=1")
    def test_wins_every_game_it_starts(self):
        cache, results = {}, []

        def explore(s, plies):
            self.assertLessEqual(plies, 13)
            moves = [cache.setdefault(s, best_move(s, INFINITE))] if s.to_move == "X" else available_moves(s)
            for c in moves:
                out = result(s, c, INFINITE)
                if out.line:
                    results.append(s.to_move)
                else:
                    explore(out.state, plies + 1)

        explore(State((), (), "X"), 0)
        self.assertEqual(set(results), {"X"})

    def test_third_repetition_loses(self):
        # O to move; playing 5 would recreate a position already seen twice.
        s = play(INFINITE, [1, 4, 0, 2, 6, 3, 8])
        bad = result(s, 5, INFINITE).state
        game = {"mode": INFINITE, "board": {"X": list(s.X), "O": list(s.O)}, "toMove": "O",
                "history": {position_key(bad): 2}}
        a = analyze(game, {"ai": "O"})
        self.assertIn(5, a["repetition"]["losingCells"])
        self.assertNotEqual(a["chosenMove"], 5)
        five = next(c for c in a["candidates"] if c["cell"] == 5)
        self.assertLessEqual(five["score"], -5)


class NormalAgentTest(unittest.TestCase):
    def test_sometimes_slips_but_never_misses_a_win_in_one(self):
        start = play(CLASSIC, [4])                      # O to move: the edges lose, the corners draw
        slips = 0
        for seed in range(40):
            a = analyze({"mode": CLASSIC, "board": {"X": list(start.X), "O": list(start.O)}, "toMove": "O"},
                        {"ai": "O", "difficulty": "normal", "seed": seed})
            slips += a["slip"]
            self.assertEqual(a["depthLimit"], 2)
        self.assertGreater(slips, 0)
        win = play(CLASSIC, [0, 3, 1, 4])
        for seed in range(20):
            a = analyze({"mode": CLASSIC, "board": {"X": list(win.X), "O": list(win.O)}, "toMove": "X"},
                        {"ai": "X", "difficulty": "normal", "seed": seed, "slipChance": 1})
            self.assertEqual(a["chosenMove"], 2)


class ContractTest(unittest.TestCase):
    UI_FIELDS = {"maxStep", "chosenMove", "depthReached", "value", "nodes", "depthLimit", "slip",
                 "optimalMove", "exact", "tree", "forcedWinIn", "candidates", "timeMs", "reason",
                 "principalVariation", "minimaxTimeMs", "deepening", "cuts", "board",
                 "vanishesNext", "repetition", "optimalValue", "forcedLossIn"}

    def check(self, a):
        self.assertTrue(self.UI_FIELDS <= a.keys(), self.UI_FIELDS - a.keys())
        self.assertLess(a["nodes"]["alphaBeta"], a["nodes"]["minimax"])
        self.assertTrue(all(-10 <= c["score"] <= 10 for c in a["candidates"]))
        self.assertEqual(a["principalVariation"][0], a["chosenMove"])

        def walk(nodes):
            for n in nodes:
                self.assertLessEqual(n["visitOrder"], n["doneOrder"])
                self.assertLessEqual(n["doneOrder"], a["maxStep"])
                self.assertEqual(n["value"] is None, n["pruned"])
                walk(n.get("children") or [])
        walk(a["tree"]["children"])
        text = json.dumps(jsonable(a), allow_nan=False)
        self.assertIn('"-inf"', text)

    def test_shapes(self):
        for mode, cells, diff in [(CLASSIC, [], "unbeatable"), (CLASSIC, [4, 0, 8], "normal"),
                                  (INFINITE, [1, 4, 0, 2, 6, 3], "unbeatable"), (INFINITE, [4], "normal")]:
            s = play(mode, cells)
            a = analyze({"mode": mode, "board": {"X": list(s.X), "O": list(s.O)}, "toMove": s.to_move},
                        {"ai": s.to_move, "difficulty": diff, "seed": 1})
            self.check(a)
            if mode == INFINITE:
                self.assertTrue(all(abs(c["score"]) >= 5 or not c["exact"] for c in a["candidates"]))

    def test_rejects_wrong_turn(self):
        with self.assertRaises(ValueError):
            analyze({"mode": CLASSIC, "board": {"X": [4], "O": []}, "toMove": "O"}, {"ai": "X"})


if __name__ == "__main__":
    unittest.main()
