import gym
import numpy as np
from gym import spaces


class Utils:
    king_moves = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
    directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]


class MateEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, board_size=4):
        super(MateEnv, self).__init__()
        self.action_space = spaces.Discrete(4 * board_size + 4)

        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(board_size, board_size), dtype=np.uint8)

        self.board_size = board_size

        self.white_king = (0, 0)
        self.white_rook = (1, 1)
        self.black_king = (2, 2)

        self.step_num = 0
        self.max_steps = 100

        self.rook_moves = []
        for j in range(self.board_size - 1):
            for i in range(4):
                self.rook_moves.append(tuple([x * (j + 1) for x in Utils.directions[i]]))

    def step(self, action, render=False):
        reward = -1
        done = self.step_num >= self.max_steps

        legal_move = False

        # If the action is less than 8, we move the white king
        if action < 8:
            new_king_position = [self.white_king[0] + Utils.king_moves[action][0],
                                 self.white_king[1] + Utils.king_moves[action][1]]

            # We need to check three things:
            # 1. The new location is on the board
            # 2. The new location is not occupied by the white rook
            # 3. The new location is not right next to the black king
            legal_move = self._is_on_board(new_king_position) and \
                         new_king_position != self.white_rook and \
                         self._manhattan_distance(new_king_position, self.black_king) > 1

            if legal_move:
                self.white_king = tuple(new_king_position)
        else:
            rook_action_index = action - 8

            new_rook_position = [self.white_rook[0] + self.rook_moves[rook_action_index][0],
                                 self.white_rook[1] + self.rook_moves[rook_action_index][1]]

            # Here, we need to check that:
            # 1. The new location is on the board, and
            # 2. The new location is not occupied by the white king, and
            # 3. The new location is not occupied by the black king
            legal_move = self._is_on_board(new_rook_position) and \
                         new_rook_position != self.white_king and \
                         new_rook_position != self.black_king

            if legal_move:
                self.white_rook = tuple(new_rook_position)

        if render:
            print("After white move:")
            self.render()

        if not legal_move:
            # Now, we need to generate all the possible moves for the black king.
            black_king_moves = [(self.black_king[0] + x[0], self.black_king[1] + x[1]) for x in Utils.king_moves]
            black_king_moves = list(filter(self._is_on_board, black_king_moves))
            black_king_moves = list(filter(lambda x: x not in self._squares_under_white_attack(), black_king_moves))

            # If there are no legal moves for the black king, the game is over.
            if len(black_king_moves) == 0:
                reward = 0 if self._checkmate() else - (self.max_steps - self.step_num)
                done = True
            else:
                # Otherwise, the black king makes the move that minimizes the manhattan distance to the white rook.
                black_king_distances = [self._manhattan_distance(x, self.white_rook) for x in black_king_moves]
                best_move = black_king_moves[np.argmin(black_king_distances)]

                # If the black king captures the white rook, the game is over.
                self.black_king = best_move

                if self.black_king == self.white_rook:
                    reward = - (self.max_steps - self.step_num)
                    done = True

            if render:
                print("After black move:")
                self.render()
        else:
            print("(tried playing an illegal move)")

        self.step_num += 1

        return self._get_board_obs(), reward, done, {}

    def _checkmate(self):
        return self.black_king in self._squares_under_white_attack()

    def _squares_under_white_attack(self):
        # Generate all the squares that are under attack by the white pieces
        white_king_attack = [(self.white_king[0] + x[0], self.white_king[1] + x[1]) for x in Utils.king_moves]
        # Filter out the squares that are off the board
        white_king_attack = list(filter(self._is_on_board, white_king_attack))

        # Generate all the squares that are under attack by the white rook
        # Starting from the white rook, go in each of the four directions until we hit the edge of the board
        # or the white king. Add each square to the list of squares under attack.
        white_rook_attack = []
        for i in range(4):
            for j in range(1, self.board_size):
                new_square = (self.white_rook[0] + j * Utils.directions[i][0],
                              self.white_rook[1] + j * Utils.directions[i][1])

                if not self._is_on_board(new_square) or new_square == self.white_king:
                    break

                white_rook_attack.append(new_square)

        return white_king_attack + white_rook_attack

    def _is_on_board(self, position):
        return 0 <= position[0] < self.board_size and 0 <= position[1] < self.board_size

    def _manhattan_distance(self, position1, position2):
        return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])

    def reset(self):
        self.white_king = (0, 0)
        self.white_rook = (1, 1)
        self.black_king = (2, 2)

        self.step_num = 0
        self.max_steps = 100

        return self._get_board_obs()

    def _get_board_obs(self):
        obs = np.zeros((self.board_size, self.board_size), dtype=np.uint8)
        obs[self.white_king[0], self.white_king[1]] = 1
        obs[self.white_rook[0], self.white_rook[1]] = 2
        obs[self.black_king[0], self.black_king[1]] = 3

        return obs

    def render(self, mode='human'):
        # Pretty-print the chess board
        board = self._get_board_obs()

        for row in board:
            print(" ".join([['.', 'K', 'R', 'k'][x] for x in row]))
        print()


    def close(self):
        pass
