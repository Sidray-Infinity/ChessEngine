import chess.pgn
import numpy as np
import os


def serialize(board_fen):
    """
    Encoding Scheme:

    [Q, K, R, B, N, P, p, n, b, r, k, q]. Capital -> White
    Returns a 768(8x8x12) bit board representation.
    """

    assert len(board_fen) != 0

    def switcher(arg):  # Helper switch function
        switcher = {
            'Q': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'K': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'R': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'B': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            'N': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            'P': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            'p': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'n': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            'b': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            'r': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            'k': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            'q': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        }
        return switcher.get(arg, "INVALID")

    def is_a_number(p):
        if p in ['1', '2', '3', '4', '5', '6', '7', '8']:
            return True
        return False

    board = board_fen.split(" ")
    states = board[0].split('/')
    final_board = []
    for state in states:
        row = []
        for p in state:
            if(is_a_number(p)):
                for i in range(int(p)):
                    row.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            else:
                row.append(switcher(p))

        final_board.append(row)

    # Returning it as a 1D array
    return np.asarray(final_board)

def from_square(move):
    from_sqr = np.zeros(64, dtype=np.int32)
    from_sqr[move.from_square] = 1
    return from_sqr

def to_square(move):
    to_sqr = np.zeros(64, dtype=np.int32)
    to_sqr[move.to_square] = 1
    return to_sqr

if __name__ == "__main__":
    # * -> Ongoing game. Chess is weird
    result = {"1/2-1/2": 0, "1-0": 1, "0-1": -1, "*": 2}
    X = []
    Y_from = []
    Y_to = []

    count = 0
    file = 0
    gn = 0
    ts = 0

    pgn = open("Data/raw.pgn")

    while True:
        game = chess.pgn.read_game(pgn)
        if game is None:
            break
        board = game.board()
        for move in game.mainline_moves():
            X.append(serialize(board.fen()))
            Y_from.append(from_square(move))
            Y_to.append(to_square(move))
            count += 1
            ts += 1
            board.push(move)

        print(f"GAME: {gn} SAMPLES: {ts}")
        gn += 1

        if count > 100000:
            print("---------------------------------------------------------------")
            print(f"Writing file : {file}")
            np.savez(f"Data/from_{file}", np.asarray(X), np.asarray(Y_from))
            np.savez(f"Data/to_{file}", np.asarray(X), np.asarray(Y_to))
            X = []
            Y_from = []
            Y_to = []
            file += 1
            count = 0

        

