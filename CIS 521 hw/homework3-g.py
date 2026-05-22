############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

import math
import itertools
import heapq
import random
import copy

############################################################

# >>>>>>>>>>>>><> Put in ID block here <><<<<<<<<<<<

############################################################
# Section 1: Tile Puzzle
############################################################
# run via % python3 homework3_tile_puzzle_gui.py rows cols


def create_tile_puzzle(rows, cols) -> "TilePuzzle":
    board = [[] for _ in range(rows)]
    for n in range(1, rows*cols):
        board[(n-1) // cols].append(n)
    board[-1].append(0)
    return TilePuzzle(board)


class TilePuzzle(object):
    # We will use the convention that tile 0 should be located
    # in the lower-right corner for the solved configuration.

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

    def perform_move(self, direction: str) -> bool:
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

    def scramble(self, num_moves) -> None:
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
        new_tile_puzzle = TilePuzzle(copy.deepcopy(self.board))  # Problem with name collision?
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
            solution_set = self.iddfs_helper(self, limit, [])  # Will come back empty if no solutions at this level
            if solution_set:
                found = True
                yield from solution_set  # find_solutions_iddfs is a generator!
            limit += 1  # If no solutions were returned, `limit` is too low.
            # We trust the puzzle given to us is solvable, so no "give up" logic is needed
    
    def iddfs_helper(self, limit, moves: list[str]) -> list[list[str]]:
        """find all solutions within the step limit based on the `moves` already taken."""
        if self.is_solved():
            return [moves]  # iddfs_helper returns all solutions found, and each solution is itself a list of moves. So its return type is list[list[str]]: a list of move-lists, not a single move-list.
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
        )  # A nice feature is that since ties break in insertion order, the search stays deterministic (FIFO)
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


def find_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    scene: list[list[bool]],
):
    """A* search using the straight-line Euclidean distance heuristic.
    To run: % python3 homework3_grid_navigation_gui.py scene_path
    The heuristic is admissible and consistent. Moving up, down, left, or
    right costs 1.0, while moving up-left, up-right, down-left, or
    down-right costs sqrt(2).
    """
    counter = itertools.count()
    priority_queue = [
        (euclid_dist(start, goal), next(counter), 0, start, None)
    ]
    # We included a counter to break ties in f_n. Nodes on the heap will
    # be of this form: f_n, next(counter), dist_so_far, (row, col), parent
    visited = dict()  # Should hold ((row, col), parent) tuples. Since
    # Euclidean distance is a *consistent (monotone)* heuristic, we can
    # mark a point visited when it is first reached/pushed.
    visited[start] = None  # `start` has no parent; all others will
    while priority_queue:
        _, _, dist_so_far, coords, parent = heapq.heappop(priority_queue)
        # dist_so_far is part of every tuple on the heapq -- it is g(n)
        if coords == goal:
            # per spec, we are to output the path as a list of its points,
            # including both `start` and `goal`, e.g. [(0, 0), (1, 0), (2, 1)]
            directions = [coords]
            while goal != start:  # Abuse of variable, but hey
                goal = visited[goal][1]    # Follow parent
                directions.append(goal)    # pointers back to
            directions.reverse()           # start, then reverse list
            return directions              # and return.
        else:
            for node in diag_neighbors(coords, scene):
                if node not in visited:
                    heapq.heappush(
                        priority_queue,
                        (f_n(dist_so_far + math.sqrt(2), node, goal), next(counter),
                         dist_so_far + math.sqrt(2), node, coords),
                    )
                    visited[node] = coords
            for node in ortho_neighbors(coords, scene):
                if node not in visited:
                    heapq.heappush(
                        priority_queue,
                        (f_n(dist_so_far + 1.0, node, goal), next(counter),
                         dist_so_far + 1.00, node, coords),
                    )
                    visited[node] = coords


def euclid_dist(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def f_n(dist_so_far, curr, goal):
    return dist_so_far + euclid_dist(curr, goal)


def diag_neighbors(point, scene: list[list[bool]]):
    x, y = point[0], point[1]
    # These helper methods do have the disadvantage that they return
    # the point we just came from
    if not scene[x + 1][y + 1]:
        yield (x + 1, y + 1)
    if not scene[x - 1][y - 1]:
        yield (x - 1, y - 1)
    if not scene[x + 1][y - 1]:
        yield (x + 1, y - 1)
    if not scene[x - 1][y + 1]:
        yield (x - 1, y + 1)


def ortho_neighbors(point, scene: list[list[bool]]):
    x, y = point[0], point[1]
    if not scene[x][y + 1]:
        yield (x, y + 1)
    if not scene[x][y - 1]:
        yield (x, y - 1)
    if not scene[x + 1][y]:
        yield (x + 1, y)
    if not scene[x - 1][y]:
        yield (x - 1, y)


############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################
# Shortest problem statement in the H.W., but not the easiest ...

def solve_distinct_disks(length, n):
    """Implement an improved version of the function from Homework
    2 using an 𝐴∗ search (rather than BFS) to find an optimal solution.
    Return any solution so long as it is of minimal length."""
    # Output will be a list of strings like ["3, 4", "1, 3", "4, 5" ...]
    # Remember both forward and backward moves may be needed
    # NOTE: I will represent the disks as 1, 2, ... n, but bear in mind that
    #  the board locations are 0, 1, ... length-1 (so that disk 1 begins
    # on spot 0, disk 2 on spot 1, etc.)

def h_n(len, n, board):
    """Devise a heuristic which is admissible (never overestimates the
     distance to the goal) but informative enough to perform efficiently
    . Consider "relaxing" the disk movement rules of this puzzle."""
    h = 0
    solved_state = [0] * (len)
    for s in range(1, n+1):
        solved_state[-s] = s
    # Iterate through board, deriving the distance from each disk to its correct final position
    for index in range(len):
        if board[index] != 0:
            h += abs(index - (len-board[index]))
    return h


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
