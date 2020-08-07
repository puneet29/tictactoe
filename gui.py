import pygame
from main import TicTacToe, QPlayer, Player
import time

clicked = False


def quitGame():
    pygame.quit()
    quit()


def textObjects(text, font, color):
    text_surface = font.render(text, True, color)
    return text_surface, text_surface.get_rect()


def button(msg, x, y, w, h, ac, ic, action=None, params=None):
    mouse = pygame.mouse.get_pos()
    click = pygame.mouse.get_pressed()

    if x+w > mouse[0] > x and y+h > mouse[1] > y:
        pygame.draw.rect(game_display, ac, (x, y, w, h))
        if click[0] == 1 and action is not None:
            if params is not None:
                action(params)
            else:
                action()
    else:
        pygame.draw.rect(game_display, ic, (x, y, w, h))

    small_text = pygame.font.SysFont('calibri', 20)
    textSurf, textRect = textObjects(msg, small_text, white)
    textRect.center = ((x+(w/2)), (y+(h/2)))
    game_display.blit(textSurf, textRect)


def gameIntro():
    intro = True
    while intro:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitGame()

        game_display.fill(dark_theme)
        largeText = pygame.font.SysFont('calibri', 115)
        TextSurf, TextRect = textObjects('Tic Tac Toe', largeText, white)
        TextRect.center = ((display_width/2), (display_height/2))
        game_display.blit(TextSurf, TextRect)

        button('Play', 100, 450, 150, 50, bright_green, green, gameLoop)
        button('Quit', 550, 450, 150, 50, bright_red, red, quitGame)

        pygame.display.update()
        clock.tick(15)


def human_move(params):
    tictactoe = params['tictactoe']
    tictactoe.update(params['position'], 'x', False)
    global clicked
    clicked = True


def terminalScreen(winning):
    if winning == 'o':
        prompt = 'Computer won!'
    elif winning == 'x':
        prompt = 'You won!'
    else:
        prompt = 'It was a tie!'

    game_display.fill(dark_theme)
    largeText = pygame.font.SysFont('calibri', 115)
    TextSurf, TextRect = textObjects(prompt, largeText, white)
    TextRect.center = ((display_width/2), (display_height/2))
    game_display.blit(TextSurf, TextRect)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitGame()

        button("Play Again", 100, 450, 150, 50, bright_green, green, gameLoop)
        button("Quit", 550, 450, 150, 50, bright_red, red, quitGame)

        pygame.display.update()
        clock.tick(15)


def gameLoop():
    time.sleep(1)
    game_exit = False
    tictactoe = TicTacToe()
    points = (((300, 25), (300, 575)), ((500, 25), (500, 575)),
              ((125, 200), (675, 200)), ((125, 400), (675, 400)))
    qplayer = QPlayer('o', q_path='pretrained_model.npy')
    curr_turn = 'o'
    player = Player()
    global clicked

    while not game_exit:
        game_display.fill(white)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                quitGame()

        if tictactoe.terminal:
            terminalScreen(tictactoe.winning)

        for start, end in points:
            pygame.draw.line(game_display, black, start, end, 8)

        if curr_turn == 'o':
            action = qplayer.choose(tictactoe, False)
            tictactoe.update(action, 'o', False)
        else:
            position = 0
            for i in [4, 204, 404]:
                for j in [104, 304, 504]:
                    if player.validMove(position, tictactoe.board):
                        button('', j, i, 192, 192, white, white, human_move,
                               {'position': position, 'tictactoe': tictactoe})
                    position += 1

        position = 0
        for i in [4, 204, 404]:
            for j in [104, 304, 504]:
                curr_pos = tictactoe.board[position]
                if curr_pos == 'o':
                    game_display.blit(circle_image, (j+49, i+49))
                elif curr_pos == 'x':
                    game_display.blit(cross_image, (j+49, i+49))
                position += 1

        if (clicked and curr_turn == 'x') or curr_turn == 'o':
            curr_turn = 'o' if curr_turn == 'x' else 'x'
            clicked = False
        pygame.display.update()
        clock.tick(15)


if __name__ == "__main__":
    pygame.init()

    display_width = 800
    display_height = 600

    icon_image = pygame.image.load('Images/Icon.jpg')
    circle_image = pygame.image.load('Images/Circle.png')
    cross_image = pygame.image.load('Images/Cross.png')

    white = (255, 255, 255)
    black = (0, 0, 0)
    dark_theme = (33, 33, 33)
    light_pink = (255, 205, 210)
    yellow = (200, 200, 0)
    red = (211, 47, 47)
    cyan = (0, 255, 255)
    green = (56, 142, 60)
    bright_red = (244, 47, 47)
    bright_green = (56, 195, 60)

    pause = False

    game_display = pygame.display.set_mode((display_width, display_height))
    pygame.display.set_caption('Tic Tac Toe')
    pygame.display.set_icon(icon_image)

    clock = pygame.time.Clock()

    gameIntro()
