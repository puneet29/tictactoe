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

    def choose(self, tictactoe):
        """Prompts user to make a choice"""
        action = int(input(f'Player {self.sign}, type your move: '))
        while(not self.validMove(action-1, tictactoe.board)):
            print('Enter a valid move')
            action = int(input(f'Player {self.sign}, type your move: '))
        return action - 1


class QPlayer(Player):
    def __init__(self, sign, q_path, init='random'):
        self.sign = sign
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
            state_q = self.q_table[hash]
            action = np.argmax(state_q)
            while(not self.validMove(action, tictactoe.board)):
                state_q[action] = -100
                action = np.argmax(state_q)
        else:
            action = np.random.randint(0, 9)
        if verbose:
            print('Thinking... Action:', action+1)
        return action


class MinMaxPlayer(Player):
    def __init__(self, sign):
        self.sign = sign
        self.maximizing = True if sign == 'o' else False

    def minimax(self, tictactoe, maximize):
        """Implement minimax algorithm"""
        SCORES = {'x': -10, 'o': 10, 0: 0}
        if tictactoe.terminal:
            return SCORES[tictactoe.winning]

        if maximize:
            best_score = -sys.maxsize
            for i in range(9):
                if self.validMove(i, tictactoe.board):
                    tictactoe.update(i, 'o', False)
                    score = self.minimax(tictactoe, False)
                    tictactoe.update(i, i+1, False)
                    best_score = max(score, best_score)
        else:
            best_score = sys.maxsize
            for i in range(9):
                if self.validMove(i, tictactoe.board):
                    tictactoe.update(i, 'x', False)
                    score = self.minimax(tictactoe, True)
                    tictactoe.update(i, i+1, False)
                    best_score = min(score, best_score)
        return best_score

    def choose(self, tictactoe, verbose=True):
        """Make a choice"""
        best_score = -sys.maxsize
        best_move = -1
        for i in range(9):
            if self.validMove(i, tictactoe.board):
                tictactoe.update(i, self.sign, False)
                score = self.minimax(tictactoe, False)
                tictactoe.update(i, i+1, False)
                if self.maximizing:
                    if score > best_score:
                        best_score = score
                        best_move = i
                else:
                    if score < best_score:
                        best_score = score
                        best_move = i
        if verbose:
            print('Thinking... Action:', best_move+1)

        # Reset the updated winning after simulation
        tictactoe.winning = 0
        return best_move


class Trainer:
    def __init__(self, player1={'q': None, 'init': 'random'},
                 player2={'q': None, 'init': 'random'}):
        self.p1_dict = player1
        self.p2_dict = player2
        self.player1 = QPlayer('o', player1['q'], player1['init'])
        if player2['q'] == 'human':
            self.player2 = HumanPlayer('x')
        else:
            self.player2 = QPlayer('x', player2['q'], player2['init'])

    def train(self, lr, discount, episodes, show_every, epsilon,
              start_epsilon_decay, end_epsilon_decay, save_every, save_path):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Epsilon decay value
        epsilon_decay_value = epsilon / \
            (end_epsilon_decay - start_epsilon_decay)

        # Rewards
        overall_rewards = []
        aggr_rewards = {'ep': [], 'avg': [], 'min': [], 'max': []}
        record_rewards = 500
        wincount = 0
        losecount = 0

        # Save the params in params.txt
        with open(f'{save_path}/params.txt', 'w+') as f:
            content = f'player1: {self.p1_dict["q"]}'
            content += f'\nplayer2: {self.p2_dict["q"]}'
            content += f'\ninit1: {self.p1_dict["init"]}'
            content += f'\ninit2: {self.p2_dict["init"]}'
            content += f'\nlr: {lr}\ndiscount: {discount}'
            content += f'\nepisodes: {episodes}'
            content += f'\nshow_every: {show_every}\nepsilon: {epsilon}'
            content += f'\nstart_epsilon_decay: {start_epsilon_decay}'
            content += f'\nend_epsilon_decay: {end_epsilon_decay}'
            content += f'\nsave_every: {save_every}\nsave_path: {save_path}'
            f.write(content)

        for episode in range(episodes):
            temp = self.player1.q_table.sum(axis=1)
            print(episode, ':',
                  (len(temp) - len(temp[temp == 0])) / len(temp) * 100,
                  '%', end='\r')
            # Get initial state
            tictactoe = TicTacToe('cpu', 'cpu')
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
                    print('Training', action+1)

                # Make the new action
                new_state, reward, done, winning = tictactoe.update(
                    action, self.player1.sign, render)

                episode_reward += reward

                if not done:
                    # Update the q-table
                    max_future_q = np.max(self.player1.q_table[new_state])
                    current_q = self.player1.q_table[state, action]
                    new_q = (1-lr) * current_q + lr * \
                        (reward + discount * max_future_q)
                    self.player1.q_table[state, action] = new_q

                    # Player 2 turn
                    if self.p2_dict['q'] is None:
                        # If q2 path is None, i.e using random agent
                        action = np.random.randint(0, 9)
                        while not self.player1.validMove(action,
                                                         tictactoe.board):
                            action = np.random.randint(0, 9)
                    elif self.p2_dict['q'] == 'human':
                        # If player 2 is a human, let the human choose the
                        # action
                        action = self.player2.choose(
                            tictactoe.board, tictactoe.getStateHash())
                    else:
                        # If the q2 path is not empty, let the trained player 2
                        # choose the action
                        action = self.player2.choose(
                            tictactoe.board, tictactoe.getStateHash(),
                            False)

                    if render:
                        print('Random', action+1)
                    new_state, reward, done, winning = tictactoe.update(
                        action, self.player2.sign, render)

                # Update the new state
                state = new_state

            overall_rewards.append(episode_reward)

            # Update the aggregate rewards
            if episode % record_rewards == 0:
                # print('Avg. Lose Prob:', losecount / record_rewards * 100,
                #       '%\tAvg.Win Prob:', wincount / record_rewards * 100,
                #       '%')
                wincount = 0
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
                self.player1.q_table[state, action] = 0.5
            else:
                losecount += 1
                self.player1.q_table[state, action] = 0

            # Save the q-table to save_path
            if episode % save_every == 0:
                np.save(f'{save_path}/q_table_{episode}.npy',
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
        if player1 not in ('human', 'qagent', 'minimax'):
            raise ValueError(
                'player1 only accepts 3 values (human, qagent, minimax)')
        if player2 not in ('human', 'qagent', 'minimax'):
            raise ValueError(
                'player2 only accepts 3 values (human, qagent, minimax)')
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
        else:
            self.player1 = MinMaxPlayer('o')

        if player2 == 'human':
            self.player2 = HumanPlayer('x')
        elif player2 == 'qagent':
            self.player2 = QPlayer('x', q_path2)
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
                reward = 0.5
            elif self.winning == sign:
                reward = 1
            elif self.winning != sign:
                reward = 0
        else:
            reward = 0

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


if __name__ == "__main__":
    modes = ['train', 'evaluate', 'single_run']

    mode = modes[2]

    if mode == 'single_run':
        tictactoe = TicTacToe(player1='minimax', player2='human')
        tictactoe.play()

    elif mode == 'evaluate':
        wins = {}
        for i in range(10000):
            print(i, end='\r')
            tictactoe = TicTacToe('cpu', 'cpu',
                                  q_path1='models3/q_table_300000.npy')
            win = tictactoe.play(verbose=False)
            if win in wins:
                wins[win] += 1
            else:
                wins[win] = 1
        for i, k in wins.items():
            del wins[i]
            if i == 0:
                i = 'Tie'
            elif i == 'o':
                i = 'Win'
            else:
                i = 'Lose'
            wins[i] = k/100
        with open('models3/results_random.txt', 'w+') as f:
            f.write('Results of 10000 games against random\
agent in percentage:\n')
            f.write(str(wins))

    else:
        for i, init in zip(range(2, 3), ['random', 'zeros', 'ones']):
            print("Current model: ", i, end='\r')
            trainer = Trainer(
                player1={'q': f'models{i}/q_table_300000.npy', 'init': None})
            trainer.train(lr=0.9, discount=0.95, episodes=3_00_001,
                          show_every=3_00_000, epsilon=1,
                          start_epsilon_decay=1,
                          end_epsilon_decay=2_50_000, save_every=5_000,
                          save_path=f'models{i}')
            visualize_q_table(dirname=f'models{i}',
                              high=3_00_000,
                              iterate=5_000)
