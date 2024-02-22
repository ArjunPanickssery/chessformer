import subprocess
import random


def generate_random_position():
    # Simplified random position generator, same as before
    pieces = ['K', 'R', 'k']  # White King, White Rook, Black King
    empty = '1'  # Denotes an empty square
    board = [[empty for _ in range(5)] for _ in range(5)]
    for piece in pieces:
        x, y = random.randint(0, 4), random.randint(0, 4)
        while board[x][y] != empty:
            x, y = random.randint(0, 4), random.randint(0, 4)
        board[x][y] = piece
    fen = '/'.join(''.join(row) for row in board)
    fen += " w - - 0 1"
    return fen


def find_best_move(process, move_time=2000):
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


def run_game(executable_path="/Users/dyusha/fun_things/chessformer/stockfish/Fairy-Stockfish/src/stockfish"):
    # Start the stockfish engine process
    process = subprocess.Popen(executable_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True)

    # Initialize the game with UCI and set the variant
    init_commands = [
        "uci",
        "setoption name UCI_Variant value chessformer",
        "ucinewgame"
    ]

    for command in init_commands:
        process.stdin.write(command + "\n")
        process.stdin.flush()

    # Generate a random position and start the game from there
    fen = generate_random_position()
    process.stdin.write(f"position fen {fen}\n")
    process.stdin.flush()

    game_moves = []
    while True:
        best_move = find_best_move(process)
        if best_move == "(none)":
            # Game end condition, e.g., checkmate or stalemate
            break
        game_moves.append(best_move)
        process.stdin.write(f"position fen {fen} moves {' '.join(game_moves)}\n")
        process.stdin.flush()

    process.terminate()
    return game_moves


# Example usage
if __name__ == "__main__":
    game_moves = run_game()
    print("Game moves:", ' '.join(game_moves))
