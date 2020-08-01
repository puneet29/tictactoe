import os

import matplotlib.pyplot as plt
import numpy as np


class HumanPlayer:
    def __init__(self, sign):
        self.sign = sign

    def validMove(self, action, board):
        """Checks if the action is a valid move"""
        if action not in range(1, 10):
            return False
        if board[action-1] in range(1, 10):
            return True
        return False

    def choose(self, board, hash):
        """Prompts user to make a choice"""
        action = int(input(f'Player {self.sign}, type your move: '))
        while(not self.validMove(action, board)):
            print('Enter a valid move')
            action = int(input(f'Player {self.sign}, type your move: '))
        return action - 1


class QPlayer:
    def __init__(self, sign, q_path):
        self.sign = sign
        if q_path is None:
            self.q_table = np.ones(shape=(3**9, 9))
        else:
            self.q_table = np.load(q_path)

    def validMove(self, action, board):
        """Checks if the action is a valid move"""
        if action not in range(0, 9):
            return False
        if board[action] in range(1, 10):
            return True
        return False

    def choose(self, board, hash, verbose=True):
        """Make a choice"""
        if board != [i for i in range(1, 10)]:
            state_q = self.q_table[hash]
            action = np.argmax(state_q)
            while(not self.validMove(action, board)):
                state_q[action] = -100
                action = np.argmax(state_q)
        else:
            action = np.random.randint(0, 9)
        if verbose:
            print('Thinking...')
            print(action+1)
        return action


class Trainer:
    def __init__(self, q1=None, q2=None):
        self.q1 = q1
        self.q2 = q2
        self.player1 = QPlayer('o', q1)
        if q2 == 'human':
            self.player2 = HumanPlayer('x')
        else:
            self.player2 = QPlayer('x', q2)

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
            content = f'player1: {self.q1}\nplayer2: {self.q2}'
            content += f'\nlr: {lr}\ndiscount: {discount}'
            content += f'\nepisodes: {episodes}'
            content += f'\nshow_every: {show_every}\nepsilon: {epsilon}'
            content += f'\nstart_epsilon_decay: {start_epsilon_decay}'
            content += f'\nend_epsilon_decay: {end_epsilon_decay}'
            content += f'\nsave_every: {save_every}\nsave_path: {save_path}'
            f.write(content)

        for episode in range(episodes):
            print(episode, end='\r')
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
                    if self.q2 is None:
                        # If q2 path is None, i.e using random agent
                        # Get the action from the trained player 1 itself
                        action = self.player1.choose(tictactoe.board,
                                                     tictactoe.getStateHash(),
                                                     False)
                    elif self.q2 == 'human':
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
                print('Avg. Lose Prob:', losecount / record_rewards * 100,
                      '%\tAvg.Win Prob:', wincount / record_rewards * 100, '%')
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
        plt.show()


class TicTacToe:
    def __init__(self, player1='human', player2='human',
                 q_path1=None, q_path2=None):
        if player1 not in ('human', 'cpu'):
            raise ValueError('player1 only accepts 2 values (human, cpu)')
        if player2 not in ('human', 'cpu'):
            raise ValueError('player2 only accepts 2 values (human, cpu)')
        if q_path1 is not None and player1 == 'human':
            raise ValueError('can only use q_path1 with player1=cpu')
        if q_path2 is not None and player2 == 'human':
            raise ValueError('can only use q_path2 with player2=cpu')

        self.board = [i for i in range(1, 10)]
        self.terminal = False
        self.winning = 0

        if player1 == 'human':
            self.player1 = HumanPlayer('o')
        else:
            self.player1 = QPlayer('o', q_path1)

        if player2 == 'human':
            self.player2 = HumanPlayer('x')
        else:
            self.player2 = QPlayer('x', q_path2)

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

    def play(self):
        # Initial tic tac toe state
        self.printBoard()
        while not self.terminal:
            # Player 1 turn
            action = self.player1.choose(self.board, self.getStateHash())
            self.update(action, 'o')

            if not self.terminal:
                # Player 2 turn
                action = self.player2.choose(self.board, self.getStateHash())
                self.update(action, 'x')

        # Check if someone won the game
        if self.winning == 0:
            print('It was a tie!')
        elif self.winning == 'o':
            print('Player o won the game!')
        else:
            print('Player x won the game!')


if __name__ == "__main__":
    train = False

    if not train:
        tictactoe = TicTacToe('cpu', 'human',
                              q_path1='models8-1/q_table_150000.npy')
        tictactoe.play()
    else:
        trainer = Trainer(q1='models8/q_table_150000.npy',
                          q2='human')
        trainer.train(lr=0.9, discount=0.95, episodes=150001, show_every=5000,
                      epsilon=1, start_epsilon_decay=1,
                      end_epsilon_decay=100000, save_every=10000,
                      save_path='models8-2')
