############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

import collections
import itertools
import heapq
import random
import copy

############################################################

# 

############################################################
# Section 1: Tile Puzzle
############################################################

# We will use the convention that tile 0 should be located 
# in the lower-right corner for the solved configuration.


def create_tile_puzzle(rows, cols) -> "TilePuzzle":
    board = [[] for _ in range(rows)]
    for n in range(1, rows*cols):
        board[(n-1) // cols].append(n)
    board[-1].append(0)
    return TilePuzzle(board)


class TilePuzzle(object):

    undo = {"up": "down", "down": "up", "left": "right", "right": "left"}
    
    def __init__(self, board):
        self.board = board
        self.dims = (len(board), len(board[0]))
        for r in range(self.dims[0]):
            for c in range(self.dims[1]):
                if board[r][c] == 0:
                    self.location_of_zero = (r, c)
                    break

    def get_board(self) -> list[list[int]]:
        return self.board

    def perform_move(self, direction) -> bool:
        """Valid moves are "up", "down", "left", and "right", and relative to the 0/empty space - "up", 
        for example, means slide the tile down from above (the **empty space** goes up)
        The rows and columns shall be zero-indexed starting at the UPPER LEFT, respectively"""
        row, col = self.location_of_zero[0], self.location_of_zero[1]
        # Normally I'd use a match ... case structure but I don't know what antique version of Python the auto-grader runs.
        if direction == "up":
            if self.location_of_zero[0] == 0:  # top row
                return False
            else:
                self.board[row][col], self.board[row-1][col] = self.board[row-1][col], self.board[row][col]  # Nicely Pythonic!
                self.location_of_zero = (row-1, col)
        elif direction == "down":
            if self.location_of_zero[0] == self.dims[0]-1:  # bottom row
                return False
            else:
                self.board[row][col], self.board[row+1][col] = self.board[row+1][col], self.board[row][col]
                self.location_of_zero = (row+1, col)
        elif direction == "left":
            if self.location_of_zero[1] == 0:  # leftmost column
                return False
            else:
                self.board[row][col], self.board[row][col-1] = self.board[row][col-1], self.board[row][col]
                self.location_of_zero = (row, col-1)
        elif direction == "right":
            if self.location_of_zero[1] == self.dims[1]-1:   # rightmost column
                return False
            else:
                self.board[row][col], self.board[row][col+1] = self.board[row][col+1], self.board[row][col]
                self.location_of_zero = (row, col+1)
        else: # invalid direction
            return False
        return True  # Don't forget!

    def scramble(self, num_moves):
        for _ in range(num_moves):
            move_candidate = random.choice(["up", "down", "left", "right"])
            self.perform_move(move_candidate)  # won't crash if move is invalid - just returns False

    def is_solved(self):
        rows, cols = self.dims  # Unpack to avoid typing dims[0] and dims[1] a million times
        goal_config = [[r * cols + c + 1 for c in range(cols)] for r in range(rows)]
        goal_config[rows - 1][cols - 1] = 0
        if self.board == goal_config:
            return True
        else:
            return False

    def copy(self):
        new_tile_puzzle = TilePuzzle(copy.deepcopy(self.board))
        return new_tile_puzzle

    def successors(self):
        if self.location_of_zero[0] != 0:
            up_copy = self.copy()  # not copy(self), that would call the shallow-copy method in Python's `copy` package
            up_copy.perform_move("up")  # modifies object in place
            yield "up", up_copy
        if self.location_of_zero[0] != self.dims[0]-1:
            down_copy =  self.copy()
            down_copy.perform_move("down")  # modifies object in place
            yield "down", down_copy
        if self.location_of_zero[1] != 0:
            left_copy =  self.copy()
            left_copy.perform_move("left")  # modifies object in place
            yield "left", left_copy
        if self.location_of_zero[1] != self.dims[1]-1:
            right_copy =  self.copy()
            right_copy.perform_move("right")  # modifies object in place
            yield "right", right_copy

    # Required
    def find_solutions_iddfs(self):
        """Yields *all* optimal solutions to the current board, represented as lists of moves. You may
        assume that the board is solvable. The order in which the solutions are produced is unimportant,
        as long as all optimal solutions are present in the output."""
        if self.is_solved():  # early return
            yield []
            return
        limit = 1
        found = False
        while not found:  # Note that all the optimal solutions must be at the same level
            solution_set = self.iddfs_helper(limit, [])  # Will come back empty if no solutions at this level
            if solution_set:
                found = True
                yield from solution_set  # find_solutions_iddfs is a generator!
            limit += 1  # If no solutions were returned, `limit` is too low.
            # We trust the puzzle given to us is solvable, so no "give up" logic is needed
    
    def iddfs_helper(self, limit: int, moves: list[str]):
        """find all solutions within the step limit based on the `moves` already taken."""
        if self.is_solved():
            return [moves]
        if limit == 0:
            return []   # terminal case; at depth limit
        solutions = []
        for move in ["up", "down", "left", "right"]:
            if moves and move == self.undo[moves[-1]]:
                continue  # Not worthwhile to reverse the move you just made
            self.perform_move(move)
            solutions += self.iddfs_helper(limit-1, moves+[move])  # Again, might return empty
            self.perform_move(self.undo[move])
        return solutions
    
    # Required
    def find_solution_a_star(self):
        """Yields *AN* optimal solution to the current board, represented as a list of moves"""
        counter = itertools.count()     # This is a unique tiebreaker to go as a second comparator in the priority queue, since multiple nodes will frequently have the same f_n
        def f_n(tile_puzzle, moves):
            return len(moves) + self.heuristic(tile_puzzle.board)  # g(n) + h(n), just as in the textbook
        def frozen(puzzle: TilePuzzle) -> tuple[tuple]:
            return tuple(tuple(row) for row in puzzle.board)
        visited = {frozen(self)}  # initialize visited set
        my_pri_queue = []  # For taking (eval function, tiebreaker, tile puzzle, [moves trail]) tuples
        starting_distance = f_n(self, [])
        heapq.heappush(my_pri_queue, 
            (starting_distance, next(counter), self, [])  # carry the move-list alongside each state
        )
        while my_pri_queue:
            eval_fun, _, state, move_list = heapq.heappop(my_pri_queue)
            if state.is_solved():
                return move_list
            for move, result_state in self.successors():  
                tuplify_board = frozen(result_state)
                if tuplify_board in visited:
                    continue
                else:
                    visited.add(tuplify_board)
                    updated_moves = move_list + [move]
                    heapq.heappush(my_pri_queue, 
                        (f_n(result_state, updated_moves), next(counter), result_state, updated_moves)
                    )  # tuple, with priority being f_n
            
            
    def heuristic(self, board: list[list[int]]) -> int:
        """Helper method for A*: calculates the Manhattan/taxicab distance from board state to solved state."""
        h, rows, cols = 0, len(board), len(board[0])
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 0:
                    continue
                goal_row = (board[r][c]-1) // cols
                goal_col = (board[r][c]-1) % cols
                h += abs(r - goal_row) + abs(c - goal_col)
                # Adds the vertical and horizontal components of taxicab distance, respectively, to h
        return h

############################################################
# Section 2: Grid Navigation
############################################################

"Can move up, down, left, right, up-left, up-right, down-left, or down-right"
# Your implementation should consist of an 𝐴∗ search using the straight-line Euclidean distance heuristic.
def find_path(start, goal, scene):
    pass

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    pass

############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
