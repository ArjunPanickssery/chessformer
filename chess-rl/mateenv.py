import gym
import numpy as np
from gym import spaces


class MateEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(MateEnv, self).__init__()
        self.action_space = spaces.Discrete(36)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(8, 8), dtype=np.uint8)

        self.step_num = 0

        self.board = np.zeros([8, 8])

        # Generate a random position for the white king
        white_king_position = (np.random.randint(8), np.random.randint(8))

        # Generate a random position for the white rook
        white_rook_position = (np.random.randint(8), np.random.randint(8))
        while white_rook_position == white_king_position:
            white_rook_position = (np.random.randint(8), np.random.randint(8))

        # Generate a random position for the black king
        black_king_position = (np.random.randint(8), np.random.randint(8))
        while black_king_position == white_king_position or black_king_position[0] == white_rook_position[0] or black_king_position[1] == white_rook_position[1]:
            black_king_position = (np.random.randint(8), np.random.randint(8))

        self.board[black_king_position] = 1
        self.board[white_king_position] = 2
        self.board[white_rook_position] = 3

        self.max_steps = 100

    def check_new_board_legality(self, new_board):
        # First, check that the black king, white king, and white rook are all still on the board
        future_black_king_location = np.where(new_board == 1)
        future_white_king_location = np.where(new_board == 2)
        future_white_rook_location = np.where(new_board == 3)

        if len(future_black_king_location[0]) != 1 or len(future_white_king_location[0]) != 1 or len(
                future_white_rook_location[0]) != 1:
            return False

        # Next, check that the black and white kings are not on adjacent squares (including diagonals)
        if abs(future_black_king_location[0][0] - future_white_king_location[0][0]) <= 1 and abs(
                future_black_king_location[1][0] - future_white_king_location[1][0]) <= 1:
            return False

        return True

    def step(self, action, render=False):
        # Default variables
        observation = self.board
        reward = -1
        done = self.step_num >= self.max_steps
        info = {}

        current_black_king_location = np.where(self.board == 1)
        current_white_king_location = np.where(self.board == 2)
        current_white_rook_location = np.where(self.board == 3)

        king_moves = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
        rook_directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        # If the action is less than 8, we're moving the white king
        if action < 8:
            new_king_location = (current_white_king_location[0][0] + king_moves[action][0],
                                 current_white_king_location[1][0] + king_moves[action][1])

            new_board = self.board.copy()

            legal_move = True

            # Check if the new location is off the board
            if new_king_location[0] < 0 or new_king_location[0] > 7 or new_king_location[1] < 0 or new_king_location[1] > 7:
                legal_move = False

            # Otherwise, try making the move
            if legal_move:
                new_board[current_white_king_location] = 0
                new_board[new_king_location] = 2

            legal_move = legal_move and self.check_new_board_legality(new_board)

            # Now, only if the move is legal, update the board
            if legal_move:
                self.board = new_board
        else:  # Otherwise, we're moving the white rook
            # subtract 8 from the action
            rook_action = action - 8

            # 0-6 is left by that number of squares plus one,
            # 7-13 is up by that number of squares (minus 7) plus one,
            # 14-20 is right by that number of squares (minus 14) plus one,
            # 21-27 is down by that number of squares (minus 21) plus one
            real_direction = rook_directions[rook_action // 7]
            real_distance = rook_action % 7 + 1
            rook_move_vector = (real_direction[0] * real_distance, real_direction[1] * real_distance)

            new_rook_position = (current_white_rook_location[0][0] + rook_move_vector[0],
                                 current_white_rook_location[1][0] + rook_move_vector[1])

            new_board = self.board.copy()

            legal_move = True

            # Check if the new location is off the board
            if new_rook_position[0] < 0 or new_rook_position[0] > 7 or new_rook_position[1] < 0 or new_rook_position[1] > 7:
                legal_move = False

            # Otherwise, try making the move
            if legal_move:
                new_board[current_white_rook_location] = 0
                new_board[new_rook_position] = 3

            legal_move = legal_move and self.check_new_board_legality(new_board)

            # Now, only if the move is legal, update the board
            if legal_move:
                self.board = new_board

        if render:
            print()
            print()
            print("White move:")
            self.render()

        # Update piece position variables
        current_black_king_location = np.where(self.board == 1)
        current_white_king_location = np.where(self.board == 2)
        current_white_rook_location = np.where(self.board == 3)

        # Now, the black king mounts a response
        # First, get the legal moves that the black king has
        black_king_legal_moves = []
        for move in king_moves:
            new_location = (current_black_king_location[0][0] + move[0], current_black_king_location[1][0] + move[1])
            # If the move is on the board
            if new_location[0] >= 0 and new_location[0] <= 7 and new_location[1] >= 0 and new_location[1] <= 7:
                # And it's not within one square of the white king
                if abs(new_location[0] - current_white_king_location[0][0]) > 1 or abs(
                        new_location[1] - current_white_king_location[1][0]) > 1:
                    # And it's not on one of the squares attacked by the white rook
                    # To check this, we need to generate a list of all the squares attacked by the white rook
                    # We do this by generating a list of the squares the rook can move to in each direction
                    # and stopping when we hit a piece or the edge of the board
                    rook_attack_squares = []
                    for direction in rook_directions:
                        for distance in range(1, 8):
                            new_location = (current_white_rook_location[0][0] + direction[0] * distance,
                                            current_white_rook_location[1][0] + direction[1] * distance)
                            if new_location[0] < 0 or new_location[0] > 7 or new_location[1] < 0 or new_location[1] > 7:
                                break
                            if new_location in [current_black_king_location, current_white_king_location]:
                                break
                            rook_attack_squares.append(new_location)

                    if new_location not in rook_attack_squares:
                        black_king_legal_moves.append(move)

        # Now, check if the black king is in mate
        black_king_mate = len(black_king_legal_moves) == 0

        if black_king_mate:
            done = True

        # First, check if the black king has a move that can capture the white rook
        for move in black_king_legal_moves:
            new_location = (current_black_king_location[0][0] + move[0], current_black_king_location[1][0] + move[1])
            if new_location == current_white_rook_location:
                # If so, the game is a draw, which is really bad
                reward = -(self.max_steps + 1)
                done = True

        # Otherwise, the black king makes the move that moves it closest to the white rook
        best_distance = 100
        best_move = None
        for move in black_king_legal_moves:
            new_location = (current_black_king_location[0][0] + move[0], current_black_king_location[1][0] + move[1])
            distance = abs(new_location[0] - current_white_rook_location[0][0]) + abs(
                new_location[1] - current_white_rook_location[1][0])
            if distance < best_distance:
                best_distance = distance
                best_move = move

        if best_move is not None:
            new_board = self.board.copy()
            new_board[current_black_king_location] = 0
            new_board[(current_black_king_location[0][0] + best_move[0], current_black_king_location[1][0] + best_move[1])] = 1
            if self.check_new_board_legality(new_board):
                self.board = new_board

        if render:
            print()
            print()
            print("Black move:")
            self.render()

        self.step_num += 1

        return observation, reward, done, info

    def reset(self):
        self.step_num = 0

        self.board = np.zeros([8, 8])

        # Generate a random position for the white king
        white_king_position = (np.random.randint(8), np.random.randint(8))

        # Generate a random position for the white rook
        white_rook_position = (np.random.randint(8), np.random.randint(8))
        while white_rook_position == white_king_position:
            white_rook_position = (np.random.randint(8), np.random.randint(8))

        # Generate a random position for the black king
        black_king_position = (np.random.randint(8), np.random.randint(8))
        while black_king_position == white_king_position or black_king_position[0] == white_rook_position[0] or \
                black_king_position[1] == white_rook_position[1]:
            black_king_position = (np.random.randint(8), np.random.randint(8))

        self.board[black_king_position] = 1
        self.board[white_king_position] = 2
        self.board[white_rook_position] = 3

        return self.board

    def render(self, mode='human'):
        # Define Unicode symbols for the pieces
        symbols = {0: ' ', 1: '♚', 2: '♔', 3: '♖'}
        # Print the top border of the chessboard
        print('  +' + '---+' * 8)

        for i in range(8):
            # Print the row number starting from 8 to 1
            print(f'{8 - i} |', end='')
            for j in range(8):
                # Print each cell with the appropriate symbol
                print(f' {symbols[self.board[i, j]]} |', end='')
            print()  # Newline at the end of each row

            # Print the separator between rows
            print('  +' + '---+' * 8)

        # Print the bottom border with column labels from 'a' to 'h'
        print('    ' + ' '.join(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']))

    def close(self):
        pass
