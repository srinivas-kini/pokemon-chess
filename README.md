# Raichu

- The game of Raichu is akin to chess, which is a 2 player, turn taking, zero-sum game where there may be time
  constraints.
- One way to formulate this problem is thinking of it as a search problem across a game tree, where each state can be
  thought of as the root of a subtree, with a particular score attached to it. This score is then backed up to the
  actual root (initial state) depending on whether we need to maximize or minimize the score.
- We can assume that on traversing the tree depths, a unit cost is incurred.
- Each state is represented by an ```N x N``` matrix and a score. The board is composed of certain pieces which are
  either black or white. The player can only move a piece pertaining to one color throughout the game.
- In this case, the player with the black pieces is the ```MIN_PLAYER```

```
BLACK_PICHU = "b"
BLACK_PIKACHU = "B"
BLACK_RAICHU = "$"
```

- And the player with the white pieces is the ```MAX_PLAYER```

```
WHITE_PICHU = "w"
WHITE_PIKACHU = "W"
WHITE_RAICHU = "@"
```

- Each piece, when moved, removed or replaced represents a valid set of successor states which is determined by the
  degrees of freedom the piece has. For example, a Raichu can move across all rows,columns and diagonals whereas a Pichu
  can only move 1-2 steps diagonally. These rules are encoded as below.

```VALID_MOVES = {
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
.
.
.
```



- The terminal state (game over) is reached when the board is completely devoid of either black pieces or white pieces.
  The _terminal test_ is captured in the given code snippet.

```
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
```

- The goal of the program is to determine the best possible move given a board, which will eventually lead to victory.

### Implementation, Design Decisions and Assumptions

- As mentioned before, we traverse the game-tree and then assign scores to each state in the tree based on an evaluation
  function.
- In this case, each piece on the board is assigned a weight which is directly proportional to its degrees of freedom.
  Consider the code snippet below.

```
def evaluate(
        current_board_2d, is_max
):  # returns the current score along with the winning player
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
    return material_score
```

- The evaluation function works by counting the difference between equal pieces of opposite colors and multiplying them
  by their weights. This represents the ```material_score``` of the board.
- Each state in the game tree is evaluated using this evaluation function. Since the white player has to maximize their
  ```material_score```, the state with the highest score will be chosen as the best move. In the case of the black
  player, this will be the minimum ```material_score```.
- The code snippet below goes over the implementation of the _minimax_ algorithm, which is used for tree traversal.

```
def minimax(board, depth, is_max, A, B):
    if check_win(board):
        if is_max:
            return -10000
        else:
            return 10000

    if depth == MAX_DEPTH:
        score = evaluate(board, is_max)
        return score

    if is_max:
        succ_states = get_possible_moves(board, MAX_PLAYER)
        for s in succ_states:
            A = max(A, minimax(s, depth + 1, not is_max, A, B))
            if A >= B:
                return A
        return A
    else:
        succ_states = get_possible_moves(board, MIN_PLAYER)
        for s in succ_states:
            B = min(B, minimax(s, depth + 1, not is_max, A, B))
            if A >= B:
                return B
        return B
```

- Since the search space for this problem is massive, we implement Alpha-Beta Pruning to ensure less plausible states
  are never considered for exploration. The parameters ```A``` and ```B``` represent alpha and beta respectively.
- Finally, the ```find_best_move``` method uses an iterative deepening strategy, to gradually increase the maximum
  depth, and yields successive best states.
