############################################################
# CIS 521: Homework 2
############################################################

############################################################
# Imports
############################################################

import copy
import collections
import itertools
import random
from math import factorial
from math import prod

############################################################

student_name = "David W. Nai"  # May 16, 2026

############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    # We do not reduce the set of solutions for rotation or reflection
    num_placements = prod(range(n**2, n**2 - n, -1)) // factorial(n)
    return num_placements


def num_placements_one_per_row(n):
    # We do not reduce the set of solutions for rotation or reflection here
    # either
    num_placements_one_per_rank = n**n
    # E.g. standard chessboard has 8 ranks, each of which has 8 files
    return num_placements_one_per_rank


# Takes as argument the column offsets of sequential ranks of queens,
# e.g. [2, 0, 3, 1]
def n_queens_valid(board: list[int]):
    # Note, per spec, we don't check if the placement is complete
    #  or have any contract as to the width of the board
    if len(board) != len(set(board)):  # Two queens on the same file
        return False
    else:  # clever method to check for queens along same diagonal
        n = len(board)
        shift_right = [board[k] + list(range(n))[k] for k in range(len(board))]
        # Use list comprehension for element-wise (index-by-index) addition
        shift_left = [board[p] + list(range(n, 0, -1))[p]
                      for p in range(len(board))]
        if (len(shift_right) != len(set(shift_right)) or len(shift_left)
                != len(set(shift_left))):
            return False
    return True


def n_queens_solutions(n):
    # Produce all valid N-Queens solutions (n > 3) with a depth-first search
    solutions = []  # For storage of all the valid solutions we found
    # For example, on an 8x8 board the a-file <- 0, b-file <- 1, ..., etc.
    # Usually for DFS problems we maintain a stack, but this instance is
    # easy enough that we can let the recursive function calling itself
    # serve as the stack!
    for file in sensible_continuations(n, []):
        n_queens_helper(n, [file], solutions)
    # We don't need to keep a visited-positions set because the tree structure
    # of our row-by-row approach imposes a well-ordering. A position can't
    # be generated twice.
    return solutions


def n_queens_helper(n: int, board: list[int],
                    solutions_so_far: list[list[int]]):
    if len(board) == n:
        solutions_so_far.append(board)
        return
    # else ...
    for file in sensible_continuations(n, board):
        if n_queens_valid(board + [file]):
            n_queens_helper(n, board + [file], solutions_so_far)


def sensible_continuations(n: int, board: list[int]) -> list[int]:
    ''' Returns a list of all columns that it makes sense to try for the
    next row -- don't attempt to place next queen directly or diagonally below
    the previous one '''
    if not board:  # empty list
        return list(range(n))  # all first options may yield a solution
    elif board[-1] == (n-1):  # last queen was at the right edge
        return list(range(n-2))  # don't try under or down-diagonally-left
    elif board[-1] == 0:  # last queen was at the left edge
        return list(range(2, n))  # don't try under or down-diagonally-right
    else:  # last queen somewhere in the middle
        return list(range(board[-1]-1)) + list(range(board[-1]+2, n))


############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):
    """<row> by <col> grid where each cell is a light that is on (True) or off
     (False). Each move you make toggles a light along with its four-connected
      neighbors (if they exist). The goal is to get all the lights to the
      `off` state."""

    def __init__(self, board: list[list[bool]]):
        # *Not* necessarily square!
        self.board_state = copy.deepcopy(board)
        self.rows = len(board)
        self.cols = len(board[0])

    def get_board(self):
        return self.board_state

    def perform_move(self, row, col):
        if not (row < self.rows and row >= 0 and col < self.cols and col >= 0):
            print("Error: Invalid move!")
            return self
        self.board_state[row][col] = not self.board_state[row][col]
        if row > 0:
            self.board_state[row-1][col] = not self.board_state[row-1][col]
        if row < self.rows - 1:
            self.board_state[row+1][col] = not self.board_state[row+1][col]
        if col > 0:
            self.board_state[row][col-1] = not self.board_state[row][col-1]
        if col < self.cols - 1:
            self.board_state[row][col+1] = not self.board_state[row][col+1]
        return self

    def scramble(self):
        # Scramble the board by randomly calling (or not calling)
        # perform_move on every cell
        for row in range(self.rows):
            for col in range(self.cols):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):  # Use a generator. The Python idiom is:
        return not any(cell for row in self.board_state for cell in row)

    def copy(self):
        return LightsOutPuzzle(copy.deepcopy(self.board_state))
        # Creates a **new** LightsOutPuzzle instance; changes to this one's
        # board_state must not affect the new one.

    def successors(self):
        # `yield` in the assignment PDF suggests we're to use a generator
        for row in range(self.rows):
            for col in range(self.cols):
                new_l_o_puzzle = self.copy()  # deep-copied
                new_l_o_puzzle.perform_move(row, col)
                yield (row, col), new_l_o_puzzle

    def find_solution(self):
        # Check if initial config is the goal state
        if self.is_solved():
            return []  # The empty move sequence
        moves_sequences = collections.deque()
        # Track seen configurations via set of tuples-of-tuples, which are hashable
        visited = {tuple(tuple(row) for row in self.board_state)}
        # Double loop to generate all possible first moves and their 
        # resulting board states, to seed the breadth-first search
        for row in range(self.rows):
            for col in range(self.cols):
                result: list[list[bool]] = (
                    self.copy().perform_move(row, col).get_board())
                if LightsOutPuzzle(result).is_solved():
                    return [(row, col)]
                moves_sequences.append(([(row, col)], result))

                visited.add(tuple(tuple(rw) for rw in result))
        while moves_sequences:
            curr_sequence, start_state = moves_sequences.popleft()
            for row in range(self.rows):
                for col in range(self.cols):
                    result: list[list[bool]] = (
                        LightsOutPuzzle(start_state)
                        .perform_move(row, col).get_board())
                    if tuple(tuple(row) for row in result) in visited:
                        continue  # Don't waste CPU going in circles
                    if LightsOutPuzzle(result).is_solved():
                        return curr_sequence + [(row, col)]
                    else:
                        moves_sequences.append(
                            (curr_sequence + [(row, col)], result))
                        visited.add(tuple(tuple(rw) for rw in result))
        # Queue ran dry -- we've visited every reachable configuration.
        return None  # Unsolvable initial state


def create_puzzle(rows, cols) -> LightsOutPuzzle:
    """Returns a LightsOutPuzzle of the given dims with all lights off."""
    all_off = [[False for _ in range(cols)] for _ in range(rows)]
    return LightsOutPuzzle(all_off)


############################################################
# Section 3: Linear Disk Movement
############################################################


def solve_identical_disks(length, n):
    """At the beginning, spots 0 through n-1 are occupied; length > n. The
    board state is concisely represented as a bitstring, with spot i
     corresponding to the 2^(length-i-1) bit."""
    free_spots = length - n
    move_seq = []  # This will hold the "sequence of moves" breadcrumb trail
    # Moves are represented as in the PDF: (<start>, <end>)
    frontier = collections.deque()
    # queue to hold (sequence of moves, board state) tuples
    board_state = (2 ** length - 1) - (2 ** free_spots - 1)
    # Python ints are arbitrary precision so there's no overflow worry

    # Start it off with a tuple, first element of which is empty list
    frontier.append((move_seq, board_state))  # members have to be hashable
    visited = {board_state}  # holds board states we've seen before
    while (frontier):
        curr_seq, curr_state = frontier.popleft()
        for (move, result) in ident_disks_helper(length, curr_state):
            if result == 2 ** n - 1:
                return curr_seq + [move]  # It's solved!
            elif result in visited:
                continue  # This is all-important; **don't go backwards**
            else:
                frontier.append((curr_seq + [move], result))
                visited.add(result)


def ident_disks_helper(length, board_state):
    """
    Generates all moves in a given position, along with their results.
    Slot/index:    0       1       2         ...         length-1
    Bit magnitude: MSB (2^(length-1))                   LSB (2^0)
    """
    for i in range(0, length):
        # We iterate over empty spaces, which is easier than iterating
        # over disks:
        if board_state & (1 << i) != 0:
            continue
        # Guards are in place to handle the L and R edges of board correctly
        # shift from right is possible
        if i >= 1 and board_state & (1 << i - 1) != 0:
            # Slot numbers are human-readable (left to right) but bit positions
            #  are from MSB to LSB
            yield ((length - i, length - i - 1),
                   board_state ^ (1 << i) ^ (1 << i - 1))
            # Slot n is i=length-n-1; offset i is slot (length-i-1)
            # a jump in from right is possible
            if i >= 2 and board_state & (1 << i - 2) != 0:
                yield ((length - i + 1, length - i - 1),
                       board_state ^ (1 << i) ^ (1 << i - 2))
        # shift from left is possible
        if i <= length - 2 and board_state & (1 << i + 1) != 0:
            yield ((length - i - 2, length - i - 1),
                   board_state ^ (1 << i) ^ (1 << i + 1))
            # a jump in from left is possible
            if i <= length - 3 and board_state & (1 << i + 2) != 0:
                yield ((length - i - 3, length - i - 1),
                       board_state ^ (1 << i) ^ (1 << i + 2))


def solve_distinct_disks(length, n):
    board_state = [x for x in range(n)] + [None]*(length-n)

    # Disks [0, 1, ..., n-1, None, ..., None] to begin with. End goal is
    # board position [None, ..., None, n-1, n-2, ..., 0].
    solved_state = tuple([None]*(length-n) + [y for y in range(n-1, -1, -1)])
    queue = collections.deque()  
    # Built to hold tuples of (move-sequence, board-position)
    queue.append(([], tuple(board_state)))  # Starting position
    # Get into the habit of casting mutable list to immutable tuple
    seen: set[tuple] = {tuple(board_state)}  # All visited board positions
    while queue:
        move_seq, board_state = queue.popleft()  # We've already established
        # that board_state is not the solved state, or we would've returned
        # before enqueueing.
        for move, result_state in distinct_disks_helper(length, board_state):
            if result_state == solved_state:
                return move_seq + [move]  # The *whole* chain of moves
            if result_state not in seen:
                seen.add(result_state)
                queue.append((move_seq + [move], result_state))
            # otherwise, ignore since we've just found a loop


def distinct_disks_helper(length, board_state):
    """Generate all legal moves in a given position, with their results."""
    for index in range(length):
        if board_state[index] is not None:
            continue  # we only generate moves *into* empty cells
        # Takes a lot of labor to handle the left and right edges of the board
        # — but a bounds check on each move type makes the edges fall out
        # for free.

        # yield possible moves into cell from the left
        if index >= 1 and board_state[index-1] is not None:
            board_copy = list(board_state)
            board_copy[index] = board_copy[index-1]
            board_copy[index-1] = None
            yield (index-1, index), tuple(board_copy)
            # Now, since we know L neighbor's nonempty, check if jump possible
            if index >= 2 and board_state[index-2] is not None:
                board_copy = list(board_state)
                board_copy[index] = board_copy[index-2]
                board_copy[index-2] = None
                yield (index-2, index), tuple(board_copy)
        # yield possible moves into cell from right
        if index <= length-2 and board_state[index+1] is not None:
            board_copy = list(board_state)
            board_copy[index] = board_copy[index+1]
            board_copy[index+1] = None
            yield (index+1, index), tuple(board_copy)
            # Since we know R neighbor's nonempty, check if a jump is possible
            if index <= length-3 and board_state[index+2] is not None:
                board_copy = list(board_state)
                board_copy[index] = board_copy[index+2]
                board_copy[index+2] = None
                yield (index+2, index), tuple(board_copy)


############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
6 hours.
"""

feedback_question_2 = """
Most challenging was the coding -- dealing with lists of lists, and sets 
containing tuples of tuples, actually gave me nausea. Conforming to arbitrary
PEP 8 rules was also a pain.
"""

feedback_question_3 = """
I found the assignment tiring, but I am glad to have completed these coding
exercises; I feel somewhat smart now!
"""
