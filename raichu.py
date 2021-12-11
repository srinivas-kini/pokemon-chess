#
# raichu.py : Play the game of Raichu
#
# Based on skeleton code by D. Crandall, Oct 2021
#

import copy
import sys
import numpy as np


# GAME RULES (KEEP THESE GLOBAL AND FINAL) !!
WHITE_PICHU = "w"
WHITE_PIKACHU = "W"
WHITE_RAICHU = "@"
WHITE_PIECES = {WHITE_PICHU, WHITE_PIKACHU, WHITE_RAICHU}

BLACK_PICHU = "b"
BLACK_PIKACHU = "B"
BLACK_RAICHU = "$"
BLACK_PIECES = {BLACK_PICHU, BLACK_PIKACHU, BLACK_RAICHU}

EMPTY = "."

VALID_MOVES = {
    WHITE_PICHU: ((1, -1), (1, 1)),  # move diagnally down
    WHITE_PIKACHU: (
        (1, 0),
        (0, -1),
        (0, 1),
    ),  # move down, left, right
    BLACK_PICHU: ((-1, -1), (-1, 1)),  # move diagnally up
    BLACK_PIKACHU: (
        (-1, 0),
        (0, -1),
        (0, 1),
    ),  # move up, left, right
}

CAN_BE_CAPTURED_BY = {
    WHITE_PICHU: {BLACK_PICHU},
    BLACK_PICHU: {WHITE_PICHU},
    WHITE_PIKACHU: {BLACK_PICHU, BLACK_PIKACHU},
    BLACK_PIKACHU: {WHITE_PICHU, WHITE_PIKACHU},
    WHITE_RAICHU: BLACK_PIECES,
    BLACK_RAICHU: WHITE_PIECES,
}

MAX_PLAYER = "w"
MIN_PLAYER = "b"


def board_to_string(board, N):
    return "\n".join(board[i : i + N] for i in range(0, len(board), N))


def make_1d(board_2d):
    return "".join(np.ndarray.flatten(board_2d))


def make_2d(board_1d, N):
    return np.reshape((list(board_1d)), (N, N))


def in_bounds(N, r, c):
    return True if 0 <= r < N and 0 <= c < N else False


def get_possible_moves(curr_board_2d, player):
    moves = []
    piece_type = WHITE_PIECES if player == MAX_PLAYER else BLACK_PIECES

    for r in range(len(curr_board_2d)):
        for c in range(len(curr_board_2d[0])):
            if curr_board_2d[r][c] in piece_type:
                moves.extend(valid_moves(curr_board_2d, r, c))
    return moves


def valid_moves(curr_board_2d, row, col):

    succ_boards = []
    if curr_board_2d[row][col] == WHITE_PICHU:
        for move in VALID_MOVES[WHITE_PICHU]:
            board = copy.deepcopy(curr_board_2d)
            if in_bounds(len(board), row + move[0], col + move[1]):
                if board[row + move[0]][col + move[1]] == EMPTY:
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = (
                        WHITE_RAICHU if row + move[0] == len(board) - 1 else WHITE_PICHU
                    )
                    succ_boards.append(board)
                elif (
                    in_bounds(len(board), row + move[0] * 2, col + move[1] * 2)
                    and board[row + move[0] * 2][col + move[1] * 2] == EMPTY
                    and board[row + move[0]][col + move[1]] == BLACK_PICHU
                ):
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = EMPTY
                    board[row + move[0] * 2][col + move[1] * 2] = (
                        WHITE_RAICHU
                        if row + move[0] * 2 == len(board) - 1
                        else WHITE_PICHU
                    )
                    succ_boards.append(board)

    elif curr_board_2d[row][col] == BLACK_PICHU:
        for move in VALID_MOVES[BLACK_PICHU]:
            board = copy.deepcopy(curr_board_2d)
            if in_bounds(len(board), row + move[0], col + move[1]):
                if board[row + move[0]][col + move[1]] == EMPTY:
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = (
                        BLACK_RAICHU if row + move[0] == 0 else BLACK_PICHU
                    )
                    succ_boards.append(board)
                elif (
                    in_bounds(len(board), row + move[0] * 2, col + move[1] * 2)
                    and board[row + move[0] * 2][col + move[1] * 2] == EMPTY
                    and board[row + move[0]][col + move[1]] == WHITE_PICHU
                ):
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = EMPTY
                    board[row + move[0] * 2][col + move[1] * 2] = (
                        BLACK_RAICHU if row + move[0] * 2 == 0 else BLACK_PICHU
                    )
                    succ_boards.append(board)

    elif curr_board_2d[row][col] == WHITE_PIKACHU:
        for move in VALID_MOVES[WHITE_PIKACHU]:
            board = copy.deepcopy(curr_board_2d)
            if in_bounds(len(board), row + move[0], col + move[1]):
                if board[row + move[0]][col + move[1]] == EMPTY:
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = (
                        WHITE_RAICHU
                        if row + move[0] == len(board) - 1
                        else WHITE_PIKACHU
                    )
                    succ_boards.append(board)

                    board = copy.deepcopy(curr_board_2d)
                    if in_bounds(len(board), row + move[0] * 2, col + move[1] * 2):
                        if board[row + move[0] * 2][col + move[1] * 2] == EMPTY:
                            board[row][col] = EMPTY
                            board[row + move[0] * 2][col + move[1] * 2] = (
                                WHITE_RAICHU
                                if row + move[0] * 2 == len(board) - 1
                                else WHITE_PIKACHU
                            )
                            succ_boards.append(board)
                        elif (
                            board[row + move[0] * 2][col + move[1] * 2]
                            in CAN_BE_CAPTURED_BY[WHITE_PIKACHU]
                            and in_bounds(
                                len(board), row + move[0] * 3, col + move[1] * 3
                            )
                            and board[row + move[0] * 3][col + move[1] * 3] == EMPTY
                        ):
                            board[row][col] = EMPTY
                            board[row + move[0] * 3][col + move[1] * 3] = (
                                WHITE_RAICHU
                                if row + move[0] * 3 == len(board) - 1
                                else WHITE_PIKACHU
                            )
                            board[row + move[0] * 2][col + move[1] * 2] = EMPTY
                            succ_boards.append(board)

                elif (
                    board[row + move[0]][col + move[1]]
                    in CAN_BE_CAPTURED_BY[WHITE_PIKACHU]
                ):
                    if (
                        in_bounds(len(board), row + move[0] * 2, col + move[1] * 2)
                        and board[row + move[0] * 2][col + move[1] * 2] == EMPTY
                    ):

                        board[row][col] = EMPTY
                        board[row + move[0]][col + move[1]] = EMPTY
                        board[row + move[0] * 2][col + move[1] * 2] = (
                            WHITE_RAICHU
                            if row + move[0] * 2 == len(board) - 1
                            else WHITE_PIKACHU
                        )
                        succ_boards.append(board)

                        board = copy.deepcopy(curr_board_2d)
                        if (
                            in_bounds(len(board), row + move[0] * 3, col + move[1] * 3)
                            and board[row + move[0] * 3][col + move[1] * 3] == EMPTY
                        ):

                            board[row][col] = EMPTY
                            board[row + move[0]][col + move[1]] = EMPTY
                            board[row + move[0] * 3][col + move[1] * 3] = (
                                WHITE_RAICHU
                                if row + move[0] * 3 == len(board) - 1
                                else WHITE_PIKACHU
                            )
                            succ_boards.append(board)

    elif curr_board_2d[row][col] == BLACK_PIKACHU:
        for move in VALID_MOVES[BLACK_PIKACHU]:
            board = copy.deepcopy(curr_board_2d)
            if in_bounds(len(board), row + move[0], col + move[1]):
                if board[row + move[0]][col + move[1]] == EMPTY:
                    board[row][col] = EMPTY
                    board[row + move[0]][col + move[1]] = (
                        BLACK_RAICHU if row + move[0] == 0 else BLACK_PIKACHU
                    )
                    succ_boards.append(board)

                    board = copy.deepcopy(curr_board_2d)
                    if in_bounds(len(board), row + move[0] * 2, col + move[1] * 2):
                        if board[row + move[0] * 2][col + move[1] * 2] == EMPTY:
                            board[row][col] = EMPTY
                            board[row + move[0] * 2][col + move[1] * 2] = (
                                BLACK_RAICHU
                                if row + move[0] * 2 == 0
                                else BLACK_PIKACHU
                            )
                            succ_boards.append(board)
                        elif (
                            board[row + move[0] * 2][col + move[1] * 2]
                            in CAN_BE_CAPTURED_BY[BLACK_PIKACHU]
                            and in_bounds(
                                len(board), row + move[0] * 3, col + move[1] * 3
                            )
                            and board[row + move[0] * 3][col + move[1] * 3] == EMPTY
                        ):
                            board[row][col] = EMPTY
                            board[row + move[0] * 3][col + move[1] * 3] = (
                                BLACK_RAICHU if row + move[0] == 0 else BLACK_PIKACHU
                            )
                            board[row + move[0] * 2][col + move[1] * 2] = EMPTY
                            succ_boards.append(board)

                elif (
                    board[row + move[0]][col + move[1]]
                    in CAN_BE_CAPTURED_BY[BLACK_PIKACHU]
                ):
                    if (
                        in_bounds(len(board), row + move[0] * 2, col + move[1] * 2)
                        and board[row + move[0] * 2][col + move[1] * 2] == EMPTY
                    ):

                        board[row][col] = EMPTY
                        board[row + move[0]][col + move[1]] = EMPTY
                        board[row + move[0] * 2][col + move[1] * 2] = (
                            BLACK_RAICHU if row + move[0] * 2 == 0 else BLACK_PIKACHU
                        )
                        succ_boards.append(board)

                        board = copy.deepcopy(curr_board_2d)
                        if (
                            in_bounds(len(board), row + move[0] * 3, col + move[1] * 3)
                            and board[row + move[0] * 3][col + move[1] * 3] == EMPTY
                        ):

                            board[row][col] = EMPTY
                            board[row + move[0]][col + move[1]] = EMPTY
                            board[row + move[0] * 3][col + move[1] * 3] = (
                                BLACK_RAICHU
                                if row + move[0] * 3 == 0
                                else BLACK_PIKACHU
                            )
                            succ_boards.append(board)

    elif (
        curr_board_2d[row][col] == WHITE_RAICHU
        or curr_board_2d[row][col] == BLACK_RAICHU
    ):

        def same_player_piece(piece1, piece2):
            if piece1 == WHITE_RAICHU and piece2 in WHITE_PIECES:
                return True
            elif piece1 == BLACK_RAICHU and piece2 in BLACK_PIECES:
                return True
            return False

        def rival_player_piece(piece1, piece2):
            if piece1 == WHITE_RAICHU and piece2 in BLACK_PIECES:
                return True
            elif piece1 == BLACK_RAICHU and piece2 in WHITE_PIECES:
                return True
            return False

        cords = []

        block_cord = ()
        n_blocks = 0
        for i in range(len(curr_board_2d) - row - 1):
            r = row + i + 1
            c = col
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(row):
            r = row - i - 1
            c = col
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(len(curr_board_2d) - col - 1):
            r = row
            c = col + i + 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(col):
            r = row
            c = col - i - 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(min(len(curr_board_2d) - col - 1, len(curr_board_2d) - row - 1)):
            r = row + i + 1
            c = col + i + 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(min(len(curr_board_2d) - col - 1, row)):
            r = row - i - 1
            c = col + i + 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(min(col, len(curr_board_2d) - row - 1)):
            r = row + i + 1
            c = col - i - 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue
            cords.append([(r, c), block_cord])

        block_cord = ()
        n_blocks = 0
        for i in range(min(row, col)):
            r = row - i - 1
            c = col - i - 1
            if n_blocks > 1 or same_player_piece(
                curr_board_2d[row][col], curr_board_2d[r][c]
            ):
                break
            if rival_player_piece(curr_board_2d[row][col], curr_board_2d[r][c]):
                n_blocks += 1
                block_cord = (r, c)
                continue

            cords.append([(r, c), block_cord])

        for new_cord, kill_cord in cords:
            board = copy.deepcopy(curr_board_2d)
            board[row][col] = EMPTY
            if kill_cord:
                board[kill_cord[0]][kill_cord[1]] = EMPTY
            board[new_cord[0]][new_cord[1]] = (
                BLACK_RAICHU
                if curr_board_2d[row][col] == BLACK_RAICHU
                else WHITE_RAICHU
            )
            # print(new_cord, kill_cord)
            # print(board)
            # print("------")
            succ_boards.append(board)

    return succ_boards


def evaluate(
    current_board_2d,
):

    PICHU_SCORE = 5
    PIKACHU_SCORE = 25
    RAICHU_SCORE = 45

    PIECE_COUNT = {
        WHITE_PICHU: 0,
        WHITE_PIKACHU: 0,
        WHITE_RAICHU: 0,
        BLACK_PICHU: 0,
        BLACK_PIKACHU: 0,
        BLACK_RAICHU: 0,
        EMPTY: 0,
    }

    curr_board_1d = make_1d(current_board_2d)

    for piece in curr_board_1d:
        PIECE_COUNT[piece] += 1
    material_score = (
        PICHU_SCORE * (PIECE_COUNT[WHITE_PICHU] - PIECE_COUNT[BLACK_PICHU])
        + PIKACHU_SCORE * (PIECE_COUNT[WHITE_PIKACHU] - PIECE_COUNT[BLACK_PIKACHU])
        + RAICHU_SCORE * (PIECE_COUNT[WHITE_RAICHU] - PIECE_COUNT[BLACK_RAICHU])
    )

    # white_mobility = (
    #     PIECE_COUNT[WHITE_PICHU] * PICHU_MOB
    #     + PIECE_COUNT[WHITE_PIKACHU] * PIKACHU_MOB
    #     + PIECE_COUNT[WHITE_RAICHU] * RAICHU_MOB
    # )
    # black_mobility = (
    #     PIECE_COUNT[BLACK_PICHU] * PICHU_MOB
    #     + PIECE_COUNT[BLACK_PIKACHU] * PIKACHU_MOB
    #     + PIECE_COUNT[BLACK_RAICHU] * RAICHU_MOB
    # )
    # mobility_score = (white_mobility - black_mobility) * 0.5
    # mobility_score = 0
    return material_score


def check_win(board):
    n_w = 0
    n_b = 0
    for piece in make_1d(board):
        if piece in WHITE_PIECES:
            n_w += 1
        elif piece in BLACK_PIECES:
            n_b += 1
    if n_w == 0 or n_b == 0:
        return True
    return False


def minimax(board, depth, max_depth, is_max, A, B):

    if check_win(board):
        if is_max:
            return -10000
        else:
            return 10000

    if depth == max_depth:
        return evaluate(board)

    if is_max:
        succ_states = get_possible_moves(board, MAX_PLAYER)
        for s in succ_states:
            A = max(A, minimax(s, depth + 1, max_depth, not is_max, A, B))
            if A >= B:
                return A
        return A
    else:
        succ_states = get_possible_moves(board, MIN_PLAYER)
        for s in succ_states:
            B = min(B, minimax(s, depth + 1, max_depth, not is_max, A, B))
            if A >= B:
                return B
        return B


def find_best_move(board, N, player, timelimit):
    board_2d = make_2d(copy.deepcopy(board), N)
    succ_states = get_possible_moves(board_2d, player)

    max_depth = 7
    for i in range(max_depth):
        if player == MAX_PLAYER:
            yield max(
                succ_states,
                key=lambda S: minimax(S, 0, i, False, float("-inf"), float("inf")),
            )
        else:
            yield min(
                succ_states,
                key=lambda S: minimax(S, 0, i, True, float("-inf"), float("inf")),
            )


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise Exception("Usage: Raichu.py N player board timelimit")

    (_, N, player, board, timelimit) = sys.argv
    N = int(N)

    timelimit = int(timelimit)
    if player not in "wb":
        raise Exception("Invalid player.")

    if len(board) != N * N or 0 in [c in "wb.WB@$" for c in board]:
        raise Exception("Bad board string.")

    print(
        "Searching for best move for "
        + player
        + " from board state: \n"
        + board_to_string(board, N)
    )
    print("Here's what I decided:")

    for new_board in find_best_move(board, N, player, timelimit):
        print(make_1d(new_board))
