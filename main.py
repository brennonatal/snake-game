import sys
import time
import random

import pygame

from snake import Snake

pygame.init()
size = width, height = 600, 400

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (53, 192, 74)
red = (250, 0, 0)
# Setting screen
screen = pygame.display.set_mode(size)
# Defining fonts to use
playFont = pygame.font.Font('assets/fonts/SigmarOne-Regular.ttf', 30)
titleFont = pygame.font.Font('assets/fonts/OtomanopeeOne-Regular.ttf', 80)
regularFont = pygame.font.Font('assets/fonts/OpenSans-Regular.ttf', 25)
# Display background menu
bg = pygame.image.load('assets/images/bg-menu.jpeg')
bg = pygame.transform.scale(bg, size)

# Initialized as True to create a game loop
game_over = True
score = None
# Initializing snake head
snake = Snake()
# To keep track of the moves
x_move = 0
y_move = 0
# Food coordinates
food_x = round(random.randrange(0, width - snake.width) / 10) * 10
food_y = round(random.randrange(0, height - snake.height) / 10) * 10

clock = pygame.time.Clock()

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN and not game_over:
            if event.key == pygame.K_LEFT:
                x_move = -5
                y_move = 0
            elif event.key == pygame.K_RIGHT:
                x_move = 5
                y_move = 0
            elif event.key == pygame.K_UP:
                y_move = -5
                x_move = 0
            elif event.key == pygame.K_DOWN:
                y_move = 5
                x_move = 0

    screen.blit(bg, (0, 0))

    # To display the initial menu
    if game_over:
        if score != None:
            screen.fill(red)

            # Game over message
            title = titleFont.render('Game Over', True, black)
            titleRect = title.get_rect()
            titleRect.center = ((width / 2), 150)
            screen.blit(title, titleRect)

            # Draw score
            score_text = regularFont.render(
                f'You scored {score} point{"s" if score != 1 else ""}', True, black)
            scoreRect = score_text.get_rect()
            scoreRect.center = ((width / 2), 230)
            screen.blit(score_text, scoreRect)

            # Draw play again button
            playButton = pygame.Rect(
                215, 245, 170, 50)
            play = playFont.render('play again', True, black)
            playRect = play.get_rect()
            playRect.center = playButton.center
            screen.blit(play, (210, 260))

            # Check if button is clicked
            click, _, _ = pygame.mouse.get_pressed()
            if click:
                mouse = pygame.mouse.get_pos()
                if playButton.collidepoint(mouse):
                    time.sleep(0.2)
                    score = 0
                    game_over = False
        else:
            # Draw title
            title = titleFont.render('Snake game', True, black)
            titleRect = title.get_rect()
            titleRect.center = ((width / 2), 150)
            screen.blit(title, titleRect)

            # Draw play button
            playButton = pygame.Rect(
                215, 245, 170, 50)
            play = playFont.render('play', True, black)
            playRect = play.get_rect()
            playRect.center = playButton.center
            screen.blit(play, (260, 245))

            # Check if button is clicked
            click, _, _ = pygame.mouse.get_pressed()
            if click:
                mouse = pygame.mouse.get_pos()
                if playButton.collidepoint(mouse):
                    time.sleep(0.2)
                    score = 0
                    game_over = False

    else:
        screen.fill(black)

        # Draw score
        score_text = regularFont.render(f'Score: {score}', True, white)
        scoreRect = score_text.get_rect()
        scoreRect.center = (50, 20)
        screen.blit(score_text, scoreRect)

        snake.x += x_move
        snake.y += y_move
        # Draw snake
        pygame.draw.rect(
            screen, green, [snake.x, snake.y, snake.height, snake.width])
        # Draw food
        pygame.draw.circle(
            screen, red, (food_x, food_y), 4)
        pygame.display.update()

        # Checking if the snake hit the food
        if (snake.x < food_x + 8 and
            snake.x + snake.width + 4 > food_x and
            snake.y < food_y + 8 and
                snake.y + snake.height + 4 > food_y):
            # Increasing score
            score += 1
            # Food new position
            food_x = round(random.randrange(0, width - snake.width) / 10) * 10
            food_y = round(random.randrange(
                0, height - snake.height) / 10) * 10
            # Increasing snake speed
            snake.speed += 5

        if snake.x >= width or snake.x < 0 or snake.y >= height or snake.y <= 0:
            game_over = True
            # Stopping snake movement
            x_move = 0
            y_move = 0
            snake.reset()
            # New food location
            food_x = round(random.randrange(0, width - snake.width) / 10) * 10
            food_y = round(random.randrange(
                0, height - snake.height) / 10) * 10

        clock.tick(snake.speed)
    pygame.display.flip()
