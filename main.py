import pygame
import sys
import time

pygame.init()
size = width, height = 600, 400

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (53, 192, 74)

screen = pygame.display.set_mode(size)

playFont = pygame.font.Font('assets/fonts/SigmarOne-Regular.ttf', 30)
titleFont = pygame.font.Font('assets/fonts/OtomanopeeOne-Regular.ttf', 80)
regularFont = pygame.font.Font('assets/fonts/OpenSans-Regular.ttf', 60)

bg = pygame.image.load('assets/images/bg-menu.jpeg')
bg = pygame.transform.scale(bg, (600, 400))

user = None

while True:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    screen.blit(bg, (0, 0))

    # Check if the player chose to play
    if user is None:

        # Draw title
        title = titleFont.render('Snake game', True, black)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 150)
        screen.blit(title, titleRect)

        # Draw play button
        playButton = pygame.Rect(
            (width / 2.8), (height / 1.62), width / 3.5, 50)
        play = playFont.render('play', True, black)
        playRect = play.get_rect()
        playRect.center = playButton.center
        screen.blit(play, (260, 245))

        # Check if button is clicked
        click, _, _ = pygame.mouse.get_pressed()
        if click == 1:
            mouse = pygame.mouse.get_pos()
            if playButton.collidepoint(mouse):
                time.sleep(0.2)
                user = 1

    else:
        screen.fill(black)

    pygame.display.flip()
