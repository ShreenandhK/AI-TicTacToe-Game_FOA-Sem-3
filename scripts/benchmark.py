"""Compare plain Minimax and Alpha-Beta on identical positions.

    py scripts/benchmark.py            # Markdown table on stdout

Both searches run for real (no estimates). Each row reports the nodes
visited, the time taken, and checks that both return the same value.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from brain.game import CLASSIC, INFINITE, State, result  # noqa: E402
from brain.search import INF, Search, alphabeta, minimax  # noqa: E402

CASES = [
    ("Classic, empty board", CLASSIC, [], 9),
    ("Classic, after X takes the centre", CLASSIC, [4], 9),
    ("Classic, after X corner, O centre", CLASSIC, [0, 4], 9),
    ("Infinite, empty board, 6 moves deep", INFINITE, [], 6),
    ("Infinite, empty board, 8 moves deep", INFINITE, [], 8),
    ("Infinite, mid-game, 10 moves deep", INFINITE, [1, 4, 0, 2, 6, 3], 10),
]


def play(mode, cells):
    s = State((), (), "X")
    for c in cells:
        s = result(s, c, mode).state
    return s


def timed(fn, *args):
    t = time.perf_counter()
    value = fn(*args)
    return value, (time.perf_counter() - t) * 1000


def main():
    print("| Position | Depth | Minimax nodes | Alpha-Beta nodes | Nodes saved | Minimax ms | Alpha-Beta ms | Same value |")
    print("|---|---:|---:|---:|---:|---:|---:|:---:|")
    for label, mode, cells, depth in CASES:
        s = play(mode, cells)
        mm, ab = Search(mode, s.to_move, depth), Search(mode, s.to_move, depth)
        v_mm, t_mm = timed(minimax, s, 0, mm)
        v_ab, t_ab = timed(alphabeta, s, 0, -INF, INF, ab)
        saved = 100 * (1 - ab.nodes / mm.nodes)
        print(f"| {label} | {depth} | {mm.nodes:,} | {ab.nodes:,} | {saved:.1f}% "
              f"| {t_mm:,.0f} | {t_ab:,.1f} | {'yes' if v_mm == v_ab else 'NO'} |")


if __name__ == "__main__":
    main()
