from flask import Flask, jsonify, render_template, request
import chess

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("chess.html")


@app.route("/move", methods=["POST"])
def move():
    data = request.get_json()
    board = chess.Board(data["fen"])
    move = chess.Move.from_uci(data["move"])
    if move in board.legal_moves:
        board.push(move)
        return jsonify({"fen": board.fen()})
    return jsonify({"error": "Illegal move"}), 400


@app.route("/load_game", methods=["POST"])
def load_game():
    data = request.get_json()
    fen = data["fen"]
    moves = data["moves"]
    return jsonify({"fen": fen, "moves": moves})


@app.route("/update_game", methods=["POST"])
def update_game():
    data = request.get_json()
    fen = data["fen"]
    moves = data["moves"]
    # Broadcast this update to the web viewer, potentially using WebSocket or similar
    # For now, just return the data
    return jsonify({"fen": fen, "moves": moves})


if __name__ == "__main__":
    app.run(debug=True)
