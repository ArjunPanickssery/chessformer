import chess
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

        self.board = chess.Board('4k3/8/8/8/8/8/8/R3K3 w Q - 0 1')

        self.max_steps = 100

    def step(self, action, render=False):
        reward = -1
        done = self.step_num >= self.max_steps

        map_piece = {v: k for k, v in self.board.piece_map().items()}
        black_king_index = map_piece[chess.Piece.from_symbol('k')]
        white_king_index = map_piece[chess.Piece.from_symbol('K')]
        white_rook_index = map_piece[chess.Piece.from_symbol('R')]

        black_king_position = (black_king_index // 8, black_king_index % 8)
        white_king_position = (white_king_index // 8, white_king_index % 8)
        white_rook_position = (white_rook_index // 8, white_rook_index % 8)

        king_moves = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
        rook_directions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        # If the action is less than 8, we're moving the white king
        if action < 8:
            new_king_position = (white_king_position[0] + king_moves[action][0],
                                 white_king_position[1] + king_moves[action][1])

            legal_move = True

            # Check if the new location is off the board
            if new_king_position[0] < 0 or new_king_position[0] > 7 or new_king_position[1] < 0 or new_king_position[
                1] > 7:
                legal_move = False

            if legal_move:
                # Otherwise, convert it to uci notation
                uci0 = chess.square_name(8 * white_king_position[0] + white_king_position[1])
                uci1 = chess.square_name(8 * new_king_position[0] + new_king_position[1])
                move_uci = uci0 + uci1

                if not chess.Move.from_uci(move_uci) in self.board.legal_moves:
                    legal_move = False

            # If everything is legal, make the move
            if legal_move:
                self.board.push_uci(move_uci)
            else:
                reward = - (self.max_steps - self.step_num)
                done = True
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

            new_rook_position = (white_rook_position[0] + rook_move_vector[0],
                                 white_rook_position[1] + rook_move_vector[1])

            legal_move = True

            # Check if the new location is off the board
            if new_rook_position[0] < 0 or new_rook_position[0] > 7 or new_rook_position[1] < 0 or new_rook_position[
                1] > 7:
                legal_move = False

            # Otherwise, convert it to uci notation
            if legal_move:
                uci0 = chess.square_name(8 * white_rook_position[0] + white_rook_position[1])
                uci1 = chess.square_name(8 * new_rook_position[0] + new_rook_position[1])
                move_uci = uci0 + uci1

                if not chess.Move.from_uci(move_uci) in self.board.legal_moves:
                    legal_move = False

            # If everything is legal, make the move
            if legal_move:
                self.board.push_uci(move_uci)
            else:
                reward = - (self.max_steps - self.step_num)
                done = True

        if render:
            print()
            print()
            print("White move:")
            self.render()

        # Check if this is mate
        if self.board.is_checkmate():
            reward = 0
            done = True

        if self.board.is_stalemate():
            reward = -1
            done = True

        if not done:
            # Now, the black king mounts a response
            black_king_legal_moves = self.board.legal_moves
            move_to_take = list(black_king_legal_moves)[np.random.randint(0, len(list(black_king_legal_moves)))]

            # If we captured the white rook, the game is over
            if self.board.is_capture(move_to_take):
                reward = - (self.max_steps - self.step_num)
                done = True
            else:  # Otherwise, make the move
                self.board.push(move_to_take)

        if render:
            print()
            print()
            print("Black move:")
            self.render()

        self.step_num += 1

        return self._get_board_obs(), reward, done, {}

    def reset(self):
        self.step_num = 0

        self.step_num = 0

        self.board = chess.Board('4k3/8/8/8/8/8/8/R3K3 w Q - 0 1')

        self.max_steps = 100

        return self._get_board_obs()

    def _get_board_obs(self):
        return [[{'.': 0, 'k': 1, 'K': 2, 'R': 3}[piece] for piece in row.split(' ')]
                for row in self.board.__str__().split('\n')]

    def render(self, mode='human'):
        print(self.board)

    def close(self):
        pass
