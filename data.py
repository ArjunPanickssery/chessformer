import os
import json
import re


def save_to_json(dictionary, file_name):
    # Create directory if not present
    directory = os.path.dirname(file_name)
    if directory != "" and not os.path.exists(directory):
        os.makedirs(directory)

    with open(file_name, "w") as f:
        json.dump(dictionary, f)


def load_from_json(file_name) -> dict:
    with open(file_name, "r") as f:
        return json.load(f)


def combine_adjacent_digits(s):
    def replacer(match):
        # Sum the digits instead of counting them
        return str(sum(int(digit) for digit in match.group()))

    return re.sub(r"\d+", replacer, s)


def five_to_eight(fen):
    fen = fen.replace("/", "3/")
    if " " in fen:
        fen = fen.split(" ")[0] + "3/8/8/8 " + fen.split(" ")[1]
    else:
        fen = fen + "3/8/8/8"
    fen = combine_adjacent_digits(fen)
    return fen


def add_start_pos_suffix(fen):
    return fen + " w - - 0 1"
