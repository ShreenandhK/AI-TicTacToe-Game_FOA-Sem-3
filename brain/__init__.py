"""The AI Tic-Tac-Toe brain: Minimax with Alpha-Beta pruning (Group 5).

Python 3 standard library only. The UI calls analyze(); see README.md.
"""
from .agent import analyze, best_move
from .game import CLASSIC, INFINITE, State, initial_state

__all__ = ["analyze", "best_move", "CLASSIC", "INFINITE", "State", "initial_state"]
