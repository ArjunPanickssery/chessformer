import requests


def update_game_viewer(fen, moves):
    url = "http://127.0.0.1:5000/update_game"
    data = {"fen": fen, "moves": moves}
    response = requests.post(url, json=data)
    return response.json()


# Example usage
fen = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"  # Starting position FEN
)
moves = ["e2e4", "e7e5", "g1f3", "b8c6"]  # Sample list of UCI moves
print(update_game_viewer(fen, moves))
