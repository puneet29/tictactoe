import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from visualize import visualize_q_table


class Player:
    def validMove(self, action, board):
        """Checks if the action is a valid move"""
        if action not in range(9):
            return False
        if board[action] in range(1, 10):
            return True
        return False


class HumanPlayer(Player):
    def __init__(self, sign):
        self.sign = sign
        self.name = 'human'

    def choose(self, tictactoe):
        """Prompts user to make a choice"""
        action = int(input(f'Type your move: '))
        while(not self.validMove(action-1, tictactoe.board)):
            print('Enter a valid move')
            action = int(input(f'Type your move: '))
        return action - 1


class QPlayer(Player):
    def __init__(self, sign, q_path, init='random'):
        self.sign = sign
        self.q_path = q_path
        self.init = init
        self.name = 'qagent'

        if q_path is None:
            if init == 'random':
                self.q_table = np.random.uniform(low=0, high=1, size=(3**9, 9))
            elif init == 'zeros':
                self.q_table = np.zeros(shape=(3**9, 9))
            elif init == 'ones':
                self.q_table = np.ones(shape=(3**9, 9))
            else:
                raise ValueError('init can only be (random, zeros, or ones)')
        else:
            self.q_table = np.load(q_path)

    def choose(self, tictactoe, verbose=True):
        """Make a choice"""
        if tictactoe.board != [i for i in range(1, 10)]:
            curr_hash = tictactoe.getStateHash()
            state_q = self.q_table[curr_hash]
            best_moves = np.argwhere(state_q == np.amax(state_q))
            best_moves = best_moves.flatten().tolist()
            action = np.random.choice(best_moves)
            while(not self.validMove(action, tictactoe.board)):
                state_q[action] = -100
                best_moves = np.argwhere(state_q == np.amax(state_q))
                best_moves = best_moves.flatten().tolist()
                action = np.random.choice(best_moves)
        else:
            action = np.random.randint(0, 9)
        if verbose:
            print('Thinking... Action:', action+1)
        return action


class RandomPlayer(Player):
    def __init__(self, sign):
        self.sign = sign
        self.name = 'random'

    def choose(self, tictactoe, verbose=True):
        """Make a choice"""
        action = np.random.randint(0, 9)
        while not self.validMove(action, tictactoe.board):
            action = np.random.randint(0, 9)
        if verbose:
            print('Thinking... Action:', action+1)
        return action


class MinMaxPlayer(Player):
    def __init__(self, sign):
        self.sign = sign
        self.opponent = 'x' if sign == 'o' else 'o'
        self.name = 'minimax'
        if not os.path.exists('cache'):
            os.makedirs('cache')
        self.cache_path = 'cache/minimax.npy'
        if not os.path.exists(self.cache_path):
            self.cache = np.zeros(shape=(3**9, 9))
        else:
            self.cache = np.load(self.cache_path)

    def minimax(self, tictactoe, depth, maximize):
        """Implement minimax algorithm"""
        if tictactoe.terminal:
            if tictactoe.winning == self.sign:
                return 10 - depth
            elif tictactoe.winning == self.opponent:
                return -10 + depth
            else:
                return 0

        if maximize:
            best_score = -sys.maxsize
            for i in range(9):
                if self.validMove(i, tictactoe.board):
                    tictactoe.update(i, self.sign, False)
                    score = self.minimax(tictactoe, depth+1, False)
                    tictactoe.update(i, i+1, False)
                    best_score = max(score, best_score)
        else:
            best_score = sys.maxsize
            for i in range(9):
                if self.validMove(i, tictactoe.board):
                    tictactoe.update(i, self.opponent, False)
                    score = self.minimax(tictactoe, depth+1, True)
                    tictactoe.update(i, i+1, False)
                    best_score = min(score, best_score)
        return best_score

    def choose(self, tictactoe, verbose=True):
        """Make a choice"""
        best_score = -sys.maxsize
        best_moves = []
        curr_hash = tictactoe.getStateHash()
        if self.cache[curr_hash].sum() == 0:
            for i in range(9):
                if self.validMove(i, tictactoe.board):
                    tictactoe.update(i, self.sign, False)
                    score = self.minimax(tictactoe, 1, False)
                    tictactoe.update(i, i+1, False)
                    if score == best_score:
                        best_moves.append(i)
                    if score > best_score:
                        best_score = score
                        best_moves = [i]
            for best_move in best_moves:
                self.cache[curr_hash][best_move] = 1
            np.save(self.cache_path, self.cache)
        else:
            best_moves = np.argwhere(
                self.cache[curr_hash] == np.amax(self.cache[curr_hash]))
            best_moves = best_moves.flatten()
        best_move = np.random.choice(best_moves)

        if verbose:
            print('Thinking... Action:', best_move+1)

        # Reset the updated winning after simulation
        tictactoe.winning = 0
        return best_move


class Trainer:
    def __init__(self, q1=None, init='random',
                 player2=None, q2=None, init2='random'):
        self.player1 = QPlayer('o', q1, init)
        if player2 == 'human':
            self.player2 = HumanPlayer('x')
        elif player2 == 'minimax':
            self.player2 = MinMaxPlayer('x')
        elif player2 == 'qagent':
            self.player2 = QPlayer('x', q2, init2)
        elif player2 == 'random':
            self.player2 = RandomPlayer('x')

    def train(self, lr, discount, episodes, show_every, epsilon,
              start_epsilon_decay, end_epsilon_decay, epsilon_decay_value,
              save_every, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Rewards
        overall_rewards = []
        aggr_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        record_rewards = 500
        wincount = 0
        losecount = 0
        tiecount = 0

        # Save the params in params.txt
        with open(f'{save_path}/params.txt', 'w+') as f:
            content = f'player1: {self.player1.name}'
            content += f'\npath1: {self.player1.q_path}'
            content += f'\ninit1: {self.player1.init}'
            content += f'\nplayer2: {self.player2.name}'
            if isinstance(self.player2, QPlayer):
                content += f'\npath2: {self.player2.q_path}'
                content += f'\ninit2: {self.player2.init}'
            content += f'\nlr: {lr}\ndiscount: {discount}'
            content += f'\nepisodes: {episodes}'
            content += f'\nshow_every: {show_every}\nepsilon: {epsilon}'
            content += f'\nepsilon_decay_value: {epsilon_decay_value}'
            content += f'\nstart_epsilon_decay: {start_epsilon_decay}'
            content += f'\nend_epsilon_decay: {end_epsilon_decay}'
            content += f'\nsave_every: {save_every}\nsave_path: {save_path}'
            f.write(content)

        for episode in range(episodes):
            print('Episode:', episode, end='\r')
            # Get initial state
            tictactoe = TicTacToe(player1=self.player1.name,
                                  player2=self.player2.name)
            state = tictactoe.getStateHash()
            done = False
            episode_reward = 0

            if episode % show_every == 0:
                render = True
                print(episode)
            else:
                render = False

            while not done:
                # Exploration - Exploitation step
                state_q = self.player1.q_table[state]
                if np.random.random() > epsilon:
                    action = np.argmax(state_q)
                    while not self.player1.validMove(action, tictactoe.board):
                        # If the action is not valid for a state, save the
                        # q-value as -100 for that state-action pair
                        state_q[action] = -100
                        action = np.argmax(state_q)
                else:
                    action = np.random.randint(0, 9)
                    while not self.player1.validMove(action, tictactoe.board):
                        # If the action is not valid for a state, save the
                        # q-value as -100 for that state-action pair
                        state_q[action] = -100
                        action = np.random.randint(0, 9)

                if render:
                    print('[Training] Player 1:', action+1)

                # Make the new action
                new_state, reward, done, winning = tictactoe.update(
                    action, self.player1.sign, render)

                episode_reward += reward

                if not done:
                    # Player 2 turn
                    if isinstance(self.player2, HumanPlayer):
                        # If player 2 is a human, let the human choose the
                        # action
                        action = self.player2.choose(tictactoe)
                    else:
                        # If player 2 is minmax player or random
                        # or q agent choose the action
                        action = self.player2.choose(tictactoe, False)

                    if render:
                        print('Player 2:', action+1)
                    new_state, reward, done, winning = tictactoe.update(
                        action, self.player2.sign, render)

                    # Update the q-table
                    max_future_q = np.max(self.player1.q_table[new_state])
                    current_q = self.player1.q_table[state, action]
                    new_q = (1-lr) * current_q + lr * \
                        (reward + discount * max_future_q)
                    self.player1.q_table[state, action] = new_q

                # Update the new state
                state = new_state

            overall_rewards.append(episode_reward)

            # Update the aggregate rewards
            if episode % record_rewards == 0:
                print('Avg. Lose Prob:', losecount / record_rewards * 100,
                      '%\tAvg. Tie Prob:', tiecount / record_rewards * 100,
                      '%\tAvg. Win Prob:', wincount / record_rewards * 100,
                      '%')
                wincount = 0
                tiecount = 0
                losecount = 0
                avg_reward = sum(
                    overall_rewards[-record_rewards:]) / record_rewards
                aggr_rewards['ep'].append(episode)
                aggr_rewards['avg'].append(avg_reward)
                aggr_rewards['min'].append(
                    min(overall_rewards[-record_rewards:]))
                aggr_rewards['max'].append(
                    max(overall_rewards[-record_rewards:]))

            # Update the q values of terminal states
            if winning == 'o':
                # print(f'We got it at {episode} episode')
                wincount += 1
                self.player1.q_table[state, action] = 1
            elif winning == 0:
                tiecount += 1
                self.player1.q_table[state, action] = 0
            else:
                losecount += 1
                self.player1.q_table[state, action] = -1

            # Save the q-table to save_path
            if episode % save_every == 0:
                np.save(f'{save_path}/{episode}.npy',
                        self.player1.q_table)

            if end_epsilon_decay >= episode >= start_epsilon_decay:
                epsilon -= epsilon_decay_value

        # Save the reward stats in matplotlib figure
        plt.plot(aggr_rewards['ep'], aggr_rewards['avg'], label='average')
        plt.plot(aggr_rewards['ep'], aggr_rewards['min'], label='minimum')
        plt.plot(aggr_rewards['ep'], aggr_rewards['max'], label='maximum')
        plt.legend(loc=4)
        plt.savefig(f'{save_path}/reward_stats.png')
        plt.clf()
        # plt.show()


class TicTacToe:
    def __init__(self, player1='human', player2='human',
                 q_path1=None, q_path2=None):
        if player1 not in ('human', 'qagent', 'minimax', 'random'):
            raise ValueError('player1 only accepts 4 values\
 (human, qagent, minimax, random)')
        if player2 not in ('human', 'qagent', 'minimax', 'random'):
            raise ValueError(
                'player2 only accepts 4 values\
 (human, qagent, minimax, random)')
        if q_path1 is not None and player1 in ('human', 'minimax'):
            raise ValueError('can only use q_path1 with player1=qagent')
        if q_path2 is not None and player2 in ('human', 'minimax'):
            raise ValueError('can only use q_path2 with player2=qagent')

        self.board = [i for i in range(1, 10)]
        self.terminal = False
        self.winning = 0

        if player1 == 'human':
            self.player1 = HumanPlayer('o')
        elif player1 == 'qagent':
            self.player1 = QPlayer('o', q_path1)
        elif player1 == 'random':
            self.player1 = RandomPlayer('o')
        else:
            self.player1 = MinMaxPlayer('o')

        if player2 == 'human':
            self.player2 = HumanPlayer('x')
        elif player2 == 'qagent':
            self.player2 = QPlayer('x', q_path2)
        elif player2 == 'random':
            self.player2 = RandomPlayer('x')
        else:
            self.player2 = MinMaxPlayer('x')

    def getStateHash(self):
        """Returns the decimal hash of the ternery tic tac toe state"""
        boardState = []
        for i in self.board:
            if i == 'x':
                boardState.append(2)
            elif i == 'o':
                boardState.append(1)
            else:
                boardState.append(0)
        power = 8
        hash = 0
        for i in boardState:
            hash += 3**(power) * i
            power -= 1
        return hash

    def getState(self):
        """Returns binary state of tic tac toe in form of
        o's, x's and empty states"""
        state = [0 for i in range(27)]
        for i in range(len(self.board)):
            if self.board[i] == 'o':
                state[i] = 1
            elif self.board[i] == 'x':
                state[9+i] = 1
            else:
                state[18+i] = 1
        return state

    def printBoard(self):
        """Prints the tic tac toe board"""
        for i in range(3):
            print('{} | {} | {}'.format(
                self.board[3*i], self.board[3*i+1], self.board[3*i+2]))
            if i != 2:
                print('---------')

    def update(self, action, sign, printBoard=True):
        """Update the state of tic tac toe"""

        # Reflect the move on board
        self.board[action] = sign

        # Check if the state is terminal state and there is no valid
        # move anymore
        if bool(set(self.board).intersection(set(range(1, 10)))):
            self.terminal = False
        else:
            self.terminal = True

        # For Minimax simulation
        self.winning = 0

        # Check the winning states
        for i in range(3):
            if self.board[3*i] == self.board[3*i+1] and\
                    self.board[3*i] == self.board[3*i+2]:
                self.winning = self.board[3*i]
                self.terminal = True
                break
            if self.board[i] == self.board[3+i] and\
                    self.board[i] == self.board[6+i]:
                self.winning = self.board[i]
                self.terminal = True
                break
        if self.board[0] == self.board[4] and\
                self.board[0] == self.board[8]:
            self.winning = self.board[0]
            self.terminal = True
        elif self.board[2] == self.board[4] and\
                self.board[2] == self.board[6]:
            self.winning = self.board[2]
            self.terminal = True

        # Print the updated tic tac toe
        if printBoard:
            self.printBoard()

        # Defining reward for each move
        if self.terminal:
            if self.winning == 0:
                reward = 0
            elif self.winning == sign:
                reward = 1
            elif self.winning != sign:
                reward = -1
        else:
            reward = -0.1

        return self.getStateHash(), reward, self.terminal, self.winning

    def play(self, verbose=True):
        # Initial tic tac toe state
        self.printBoard()
        while not self.terminal:
            # Player 1 turn
            if verbose:
                print("Player 1, play your turn")
            action = self.player1.choose(self)
            self.update(action, 'o', printBoard=verbose)

            if not self.terminal:
                # Player 2 turn
                if verbose:
                    print("Player 2, play your turn")
                action = self.player2.choose(self)
                self.update(action, 'x', printBoard=verbose)

        # Check if someone won the game
        if self.winning == 0:
            print('It was a tie!')
        elif self.winning == 'o':
            print('Player o won the game!')
        else:
            print('Player x won the game!')

        return self.winning


def evaluate(qpath, player2='minimax', rounds=10000):
    """Evaluate the model"""
    wins = {}
    for i in range(rounds):
        print(i, end='\r')
        tictactoe = TicTacToe(player1='qagent',
                              q_path1=qpath,
                              player2=player2)
        win = tictactoe.play(verbose=False)
        if win in wins:
            wins[win] += 1
        else:
            wins[win] = 1

    if 0 in wins:
        wins['Tie'] = wins.pop(0) / rounds * 100
    if 'o' in wins:
        wins['Win'] = wins.pop('o') / rounds * 100
    if 'x' in wins:
        wins['Lose'] = wins.pop('x') / rounds * 100

    with open(os.path.join(os.path.dirname(qpath),
                           f'results_{player2}.txt'), 'w+') as f:
        f.write(f'Results of {rounds} games against {player2}\
 agent in percentage:\n')
        f.write(str(wins))


if __name__ == "__main__":
    print('Select a mode:')
    print('Mode 1: Training')
    print('Mode 2: Evaluate')
    print('Mode 3: Single Run')
    mode = int(input('Enter mode number: '))

    if mode == 1:
        # Define the params for training
        LR = 0.9
        DISCOUNT = 0.95
        EPISODES = 3_00_001
        SHOW_EVERY = 3_00_000
        EPSILON = 1
        START_EPSILON_DECAY = 1
        END_EPSILON_DECAY = 2_80_000
        SAVE_EVERY = 5000

        # Calculate the uniform epsilon decay value
        epsilon_decay_value = EPSILON / \
            (END_EPSILON_DECAY - START_EPSILON_DECAY)

        for num, player2 in enumerate(['random', 'minimax']):
            for i, init in enumerate(['random', 'zeros', 'ones'], start=1):
                SAVE_PATH = f'model{num*3+i}'

                trainer = Trainer(init=init, player2=player2)
                trainer.train(lr=LR, discount=DISCOUNT, episodes=EPISODES,
                              show_every=SHOW_EVERY, epsilon=EPSILON,
                              start_epsilon_decay=START_EPSILON_DECAY,
                              end_epsilon_decay=END_EPSILON_DECAY,
                              epsilon_decay_value=epsilon_decay_value,
                              save_every=SAVE_EVERY,
                              save_path=SAVE_PATH)
                visualize_q_table(dirname=SAVE_PATH,
                                  high=EPISODES-1,
                                  iterate=SAVE_EVERY)
                evaluate(qpath=f'{SAVE_PATH}/{EPISODES-1}.npy')
    elif mode == 2:
        path = input('Enter Q-Table\'s path: ')

        valid_values = ('human', 'random', 'qagent', 'minimax')
        player2 = input(f'Enter player 2 {valid_values}: ')
        while player2 not in valid_values:
            print('Invalid input')
            player2 = input(f'Enter player 2 {valid_values}: ')

        rounds = int(input('Enter number of rounds: '))

        evaluate(qpath=path, player2=player2, rounds=rounds)
    elif mode == 3:
        valid_values = ('human', 'random', 'qagent', 'minimax')
        prompt = 'Enter {}' + str(valid_values) + ': '
        players = []
        q_paths = []

        for num in range(1, 3):
            player = input(prompt.format(f'player {num}')).lower().strip()
            while player not in valid_values:
                print('Invalid input')
                player = input(prompt.format(f'player {num}')).lower().strip()
            players.append(player)

            sub_prompt = 'Enter the path to Q-Table.\n'
            sub_prompt += 'Enter default for pretrained model.\n'
            sub_prompt += 'Enter none, for randomly initialized Q-Table: '
            if player == 'qagent':
                q_path = input(sub_prompt).lower().strip()
                if q_path == 'none':
                    q_path = None
                elif q_path == 'default':
                    q_path = 'pretrained_model.npy'
                q_paths.append(q_path)
            else:
                q_paths.append(None)

        tictactoe = TicTacToe(player1=players[0], player2=players[1],
                              q_path1=q_paths[0], q_path2=q_paths[1])
        tictactoe.play()
