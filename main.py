import random
import sys
import time

import pandas as pd
import pygame

from snake import SnakeCell
from util import ate_food, snake_collision

pygame.init()
size = width, height = 600, 400

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (53, 192, 74)
red = (250, 0, 0)

# Setting screen
screen = pygame.display.set_mode(size)

# Starting the mixer
pygame.mixer.init()

# Setting the theme song
pygame.mixer.music.load("assets/musics/Snake_III_theme_song.mp3")
pygame.mixer.music.set_volume(0.7)

# Defining fonts to use
playFont = pygame.font.Font('assets/fonts/SigmarOne-Regular.ttf', 30)
titleFont = pygame.font.Font('assets/fonts/OtomanopeeOne-Regular.ttf', 80)
regularFont = pygame.font.Font('assets/fonts/OpenSans-Regular.ttf', 25)
smallFont = pygame.font.Font('assets/fonts/OpenSans-Regular.ttf', 20)

# Display background menu
bg = pygame.image.load('assets/images/bg-menu.jpeg')
bg = pygame.transform.scale(bg, size)

# Initialized as True to create a game loop
game_over = True
score = None

# Get the highest score
best_game = 0
records = pd.read_csv('records.csv', index_col=0)
best_game = records.tail(n=1).iloc[0].score

# Initializing snake
snake_head = SnakeCell()
snake_body = [snake_head]
snake_speed = 10

# To keep track of the moves
x_move = 0
y_move = 0

# Food properties
food_x = round(random.randrange(10, width - 10) / 10) * 10
food_y = round(random.randrange(10, height - 10) / 10) * 10
food_ratio = 4

clock = pygame.time.Clock()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.KEYDOWN:
            # Key options while not playing
            if game_over:
                # Play game if ENTER is pressed
                if event.key == pygame.K_RETURN:
                    # Load theme song again
                    pygame.mixer.music.play(loops=-1, start=2.5)
                    time.sleep(0.2)
                    score = 0
                    game_over = False
                # Quit game if ESC is pressed
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
            # While playing
            else:
                # Snake controllers
                if event.key == pygame.K_LEFT and x_move == 0:
                    x_move = -snake_head.size
                    y_move = 0
                elif event.key == pygame.K_RIGHT and x_move == 0:
                    x_move = snake_head.size
                    y_move = 0
                elif event.key == pygame.K_UP and y_move == 0:
                    y_move = -snake_head.size
                    x_move = 0
                elif event.key == pygame.K_DOWN and y_move == 0:
                    y_move = snake_head.size
                    x_move = 0

    screen.blit(bg, (0, 0))

    # To display the initial menu or game over
    if game_over:
        # To ensure that the snake starts with the head only
        snake_body = [snake_head]
        
        # When game is over
        if score != None:
            # Register only best games
            if score > best_game:
                records = records.append(
                    {'score': score}, ignore_index=True)
                records.to_csv('records.csv')
                best_game = score

            # Stop music
            pygame.mixer.music.stop()

            screen.fill(red)

            # Game over message
            title = titleFont.render('Game Over', True, black)
            titleRect = title.get_rect()
            titleRect.center = ((width / 2), 130)
            screen.blit(title, titleRect)

            # Draw score
            score_text = regularFont.render(
                f'You scored {score} point{"s" if score != 1 else ""}', True, black)
            scoreRect = score_text.get_rect()
            scoreRect.center = ((width / 2), 210)
            screen.blit(score_text, scoreRect)
            
            # Draw highest score
            high_score = regularFont.render(
                f'The highest score is: {best_game}', True, black)
            scoreRect = high_score.get_rect()
            scoreRect.center = ((width / 2), 240)
            screen.blit(high_score, scoreRect)

            # Draw play again button
            playButton = pygame.Rect(
                215, 245, 170, 50)
            play = playFont.render('play again', True, black)
            playRect = play.get_rect()
            playRect.center = playButton.center
            screen.blit(play, (210, 280))

            # Check if button is clicked
            click, _, _ = pygame.mouse.get_pressed()
            if click:
                mouse = pygame.mouse.get_pos()
                if playButton.collidepoint(mouse):
                    # PLay music again
                    pygame.mixer.music.play(loops=-1, start=2.5)
                    time.sleep(0.2)
                    score = 0
                    game_over = False

        # Initial menu
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
                    pygame.mixer.music.play(loops=-1, start=2.5)
                    time.sleep(0.2)
                    score = 0
                    game_over = False

    else:
        screen.fill(black)

        # Draw score
        score_text = smallFont.render(f'Score: {score}', True, white)
        scoreRect = score_text.get_rect()
        scoreRect.center = (50, 20)
        screen.blit(score_text, scoreRect)

        # Constantly move the snake head using last movement
        snake_head = SnakeCell(
            x=snake_body[0].x + x_move, y=snake_body[0].y + y_move)
        # Always drawing a new head in the movement direction and removing the tail
        snake_body.insert(0, snake_head)
        snake_body.pop()


        if snake_collision(snake_head=snake_head, snake_body=snake_body[1:]):
            # Stop theme song
            pygame.mixer.music.stop()

            # Initial config
            game_over = True
            # Stopping snake movement
            x_move = 0
            y_move = 0
            snake_head.reset()
            # New food location
            food_x = round(random.randrange(
                10, width - snake_head.size) / 10) * 10
            food_y = round(random.randrange(
                10, height - snake_head.size) / 10) * 10

            snake_speed = 10

        # Draw snake
        for cell in snake_body:
            pygame.draw.rect(
                screen, green, [cell.x, cell.y, cell.size, cell.size])

        # Draw food
        pygame.draw.circle(
            screen, red, (food_x, food_y), food_ratio)
        pygame.display.update()

        # Checking if the snake ate the food
        if ate_food(food_x=food_x, food_y=food_y, food_ratio=food_ratio, snake_x=snake_head.x, snake_y=snake_head.y, snake_size=snake_head.size):
            # Increasing score
            score += 1

            # Increasing snake length
            tail = snake_body[-1]
            snake_body.append(tail)

            # Food new position
            food_x = round(random.randrange(
                10, width - snake_head.size) / 10) * 10
            food_y = round(random.randrange(
                10, height - snake_head.size) / 10) * 10
            # Increasing snake speed
            snake_speed += 0.5

        # To make it borderless
        # if snake.x > width:
        #     snake.x = 0
        # elif snake.x < 0:
        #     snake.x = width
        # elif snake.y > height:
        #     snake.y = 0
        # elif snake.y < 0:
        #     snake.y = height

        if (snake_head.x >= width or snake_head.x < 0 or snake_head.y >= height or snake_head.y <= 0):
            # Stop theme song
            pygame.mixer.music.stop()

            # Initial config
            game_over = True
            # Stopping snake movement
            x_move = 0
            y_move = 0
            snake_head.reset()
            # New food location
            food_x = round(random.randrange(
                10, width - snake_head.size) / 10) * 10
            food_y = round(random.randrange(
                10, height - snake_head.size) / 10) * 10

            snake_speed = 10

        clock.tick(snake_speed)
    pygame.display.flip()
