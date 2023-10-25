import math

X = "X"
O = "O"
EMPTY = None

def initial_state():
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]

def player(board):
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return O if x_count > o_count else X

def actions(board):
    actions_set = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                actions_set.add((i, j))
    return actions_set

def result(board, action):
    if action not in actions(board):
        raise ValueError("Invalid action.")
    new_board = [row.copy() for row in board]
    new_board[action[0]][action[1]] = player(board)
    return new_board

def winner(board):
    winning_lines = [
        [(0, 0), (0, 1), (0, 2)], [(1, 0), (1, 1), (1, 2)], [(2, 0), (2, 1), (2, 2)],
        [(0, 0), (1, 0), (2, 0)], [(0, 1), (1, 1), (2, 1)], [(0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 1), (2, 2)], [(0, 2), (1, 1), (2, 0)]
    ]
    for line in winning_lines:
        values = [board[i][j] for (i, j) in line]
        if values == [X, X, X]:
            return X
        elif values == [O, O, O]:
            return O
    return None

def terminal(board):
    return winner(board) is not None or not any(EMPTY in row for row in board)

def utility(board):
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

def minimax(board):
    if terminal(board):
        return None
    
    def max_value(board):
        if terminal(board):
            return utility(board)
        value = -math.inf
        for action in actions(board):
            value = max(value, min_value(result(board, action)))
        return value

    def min_value(board):
        if terminal(board):
            return utility(board)
        value = math.inf
        for action in actions(board):
            value = min(value, max_value(result(board, action)))
        return value
    
    optimal_move = None
    if player(board) == X:
        best_value = -math.inf
        for action in actions(board):
            board_value = min_value(result(board, action))
            if board_value > best_value:
                best_value = board_value
                optimal_move = action
    else:
        best_value = math.inf
        for action in actions(board):
            board_value = max_value(result(board, action))
            if board_value < best_value:
                best_value = board_value
                optimal_move = action
                
    return optimal_move
