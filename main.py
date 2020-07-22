class HumanPlayer:
    def __init__(self, sign):
        self.sign = sign

    def validMove(self, action, board):
        """Checks if the action is a valid move"""
        if action not in range(1, 10):
            return False
        if board[action-1] != 'o' and board[action-1] != 'x':
            return True
        return False

    def choose(self, board):
        """Prompts user to make a choice"""
        action = int(input(f'Player {self.sign}, type your move: '))
        while(not self.validMove(action, board)):
            print('Enter a valid move')
            action = int(input(f'Player {self.sign}, type your move: '))
        return action


class TicTacToe:
    def __init__(self, compete='human'):
        self.board = [i for i in range(1, 10)]
        self.terminal = False
        self.winning = 0
        self.player1 = HumanPlayer('o')
        if compete == 'human':
            self.player2 = HumanPlayer('x')

    def printBoard(self):
        """Prints the tic tac toe board"""
        for i in range(3):
            print('{} | {} | {}'.format(
                self.board[3*i], self.board[3*i+1], self.board[3*i+2]))
            if i != 2:
                print('---------')

    def update(self, action, sign):
        """Update the state of tic tac toe"""
        # Reflect the move on board
        self.board[action-1] = sign
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
        if self.terminal is False:
            if self.board[0] == self.board[4] and\
                    self.board[0] == self.board[8]:
                self.winning = self.board[0]
                self.terminal = True
            elif self.board[2] == self.board[4] and\
                    self.board[2] == self.board[6]:
                self.winning = self.board[2]
                self.terminal = True

        # Print the updated tic tac toe
        self.printBoard()

    def play(self):
        # Initial tic tac toe state
        self.printBoard()
        while not self.terminal:
            # Player 1 turn
            action = self.player1.choose(self.board)
            self.update(action, 'o')

            if not self.terminal:
                # Player 2 turn
                action = self.player2.choose(self.board)
                self.update(action, 'x')

        # Check if someone won the game
        if self.winning == 0:
            print('It was a tie!')
        elif self.winning == 'o':
            print('Player o won the game!')
        else:
            print('Player x won the game!')


if __name__ == "__main__":
    tictactoe = TicTacToe()
    tictactoe.play()
