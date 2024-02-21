import subprocess
import random


def generate_random_position():
    # Simplified random position generator, same as before
    pieces = ["K", "R", "k"]  # White King, White Rook, Black King
    empty = "1"  # Denotes an empty square
    board = [[empty for _ in range(5)] for _ in range(5)]
    for piece in pieces:
        x, y = random.randint(0, 4), random.randint(0, 4)
        while board[x][y] != empty:
            x, y = random.randint(0, 4), random.randint(0, 4)
        board[x][y] = piece
    fen = "/".join("".join(row) for row in board)
    for n in range(8, 0, -1):
        if "1" * n in fen:
            fen = fen.replace("1" * n, str(n))

    fen += " w - - 0 1"
    # fen = "KR3/5/5/5/4k w - - 0 1"
    # fen = "KR6/8/8/8/8/8/8/7k w - - 0 1"
    print(fen)
    return fen


def find_best_move(process, move_time=5000):
    # Send command to stockfish to find the best move with a specified thinking time
    process.stdin.write(f"go movetime {move_time}\n")
    process.stdin.flush()
    best_move = None
    while True:
        line = process.stdout.readline().strip()
        if line.startswith("bestmove"):
            best_move = line.split()[1]
            break
    return best_move


def run_game(
    executable_path="engine.exe",
):
    # Start the stockfish engine process
    process = subprocess.Popen(
        executable_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Initialize the game with UCI and set the variant
    init_commands = [
        "setoption name VariantPath value variants.ini",
        "setoption name UCI_Variant value chessformer",
        # "ucinewgame",
    ]

    for command in init_commands:
        process.stdin.write(command + "\n")
        process.stdin.flush()

    # Generate a random position and start the game from there
    fen = generate_legal_fen(5)
    process.stdin.write(f"position fen {fen}\n")
    process.stdin.flush()

    game_moves = []
    while True:
        best_move = find_best_move(process)
        print(best_move)
        if best_move == "(none)":
            # Game end condition, e.g., checkmate or stalemate
            print("Game over")
            break
        game_moves.append(best_move)
        process.stdin.write(f"position fen {fen} moves {' '.join(game_moves)}\n")
        process.stdin.flush()

    process.terminate()
    return game_moves


def best_move(fen, executable_path="engine.exe"):
    # Start the stockfish engine process
    process = subprocess.Popen(
        executable_path,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Initialize the game with UCI and set the variant
    init_commands = [
        "uci",
        "setoption name VariantPath value variants.ini",
        "setoption name UCI_Variant value chessformer",
        "ucinewgame",
    ]

    for command in init_commands:
        process.stdin.write(command + "\n")
        process.stdin.flush()

    # Generate a random position and start the game from there
    process.stdin.write(f"position fen {fen}\n")
    process.stdin.flush()

    best_move = find_best_move(process)
    if best_move == "(none)":
        print("Game over")
        return None

    process.terminate()
    return best_move


def generate_legal_fen(board_size):
    # Initialize an empty 5x5 board
    board = [["-" for _ in range(board_size)] for _ in range(board_size)]

    # Function to check if kings are placed legally
    def kings_are_legal(wk, bk):
        wk_x, wk_y = wk
        bk_x, bk_y = bk
        return abs(wk_x - bk_x) > 1 or abs(wk_y - bk_y) > 1

    # Place white king (WK) and rook (WR) randomly
    wk_pos = (random.randint(0, board_size - 1), random.randint(0, board_size - 1))
    board[wk_pos[0]][wk_pos[1]] = "K"

    while True:
        wr_pos = (random.randint(0, board_size - 1), random.randint(0, board_size - 1))
        if wr_pos != wk_pos:
            board[wr_pos[0]][wr_pos[1]] = "R"
            break

    # Place black king (BK) ensuring it's not in check and not adjacent to WK
    bk_placed = False
    while not bk_placed:
        bk_pos = (random.randint(0, board_size - 1), random.randint(0, board_size - 1))
        if bk_pos != wk_pos and bk_pos != wr_pos and kings_are_legal(wk_pos, bk_pos):
            # Ensure BK is not in check (not in same row/column as WR unless blocked by WK)
            if (
                (bk_pos[0] != wr_pos[0] and bk_pos[1] != wr_pos[1])
                or (
                    bk_pos[0] == wr_pos[0]
                    and abs(bk_pos[1] - wr_pos[1]) > 1
                    and "K" in board[bk_pos[0]]
                )
                or (
                    bk_pos[1] == wr_pos[1]
                    and abs(bk_pos[0] - wr_pos[0]) > 1
                    and "K" in [row[bk_pos[1]] for row in board]
                )
            ):
                board[bk_pos[0]][bk_pos[1]] = "k"
                bk_placed = True

    # Convert board to FEN string
    fen_rows = []
    for row in board:
        fen_row = "".join(row).replace("-", "1")
        fen_rows.append(fen_row)
    fen = "/".join(fen_rows)

    for n in range(8, 0, -1):
        if "1" * n in fen:
            fen = fen.replace("1" * n, str(n))

    fen += " w - - 0 1"
    print(fen)
    return fen


# Generate and print a legal FEN
if __name__ == "__main__":
    run_game()
    # for i in range(10):
    #     print(generate_legal_fen())
    # print(res.replace("/", "3/") + "3/8/8/8")
