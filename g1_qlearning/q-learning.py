import sys
import time
import pygame
import pygame.freetype
from pygame.locals import *
from robot import Robot

# Add colors as needed.
GREEN_COLOR = pygame.Color(0, 255, 0)
BLACK_COLOR = pygame.Color(0, 0, 0)
WHITE_COLOR = pygame.Color(255, 255, 255)

if __name__ == "__main__":
    pygame.init()
    game_font = pygame.freetype.SysFont("Comic Sans MS", 10)
    fps_clock = pygame.time.Clock()

    play_surface = pygame.display.set_mode((500, 500))
    pygame.display.set_caption('Karaktersatt Oppgave 1 DTE2602')
    simulator_speed = 10 # Adjust this value to change the speed of the visualiztion. Bigger number = more faster...

    bg_image = pygame.image.load("grid.jpg") # Loads the simplified grid image.
    #bg_image = pygame.image.load("map.jpg") # Uncomment this to load the terrain map image.

    robot = Robot() # Create a new robot.
    robot.reset_random()

    # Pygame boilerplate code.
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                running = False
                break
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.event.post(pygame.event.Event(QUIT))
                    running = False
                    break

        play_surface.fill(WHITE_COLOR) # Fill the screen with white.
        play_surface.blit(bg_image, (0, 0)) # Render the background image.

        # Render the robot over the image.
        pygame.draw.rect(play_surface, BLACK_COLOR, Rect(robot.get_x() * 70 + 69, robot.get_y() * 70 + 69, 22, 22)) # A black outline.
        pygame.draw.rect(play_surface, GREEN_COLOR, Rect(robot.get_x() * 70 + 70, robot.get_y() * 70 + 70, 20, 20)) # The robot is rendered in green, you may change this if you want.
        
        # Render values from the Q-matrix
        q_matrix = robot.get_q_matrix()
        for row in range(6):
            for col in range(6):
                up, down, left, right = q_matrix[(row, col)]
                
                game_font.render_to(play_surface, (col * 73 + 69, row * 73 + 44), str(round(up)), GREEN_COLOR if up == max(q_matrix[(row, col)]) else BLACK_COLOR)
                game_font.render_to(play_surface, (col * 73 + 69, row * 73 + 94), str(round(down)), GREEN_COLOR if down == max(q_matrix[(row, col)]) else BLACK_COLOR)
                game_font.render_to(play_surface, (col * 73 + 44, row * 73 + 69), str(round(left)), GREEN_COLOR if left == max(q_matrix[(row, col)]) else BLACK_COLOR)
                game_font.render_to(play_surface, (col * 73 + 94, row * 73 + 69), str(round(right)), GREEN_COLOR if right == max(q_matrix[(row, col)]) else BLACK_COLOR)

        # Calls related to Q-learning.
        if robot.has_reached_goal():
            robot.reset_random()
        else:
            robot.one_step_q_learning()

        # Refresh the screen.
        pygame.display.flip()
        fps_clock.tick(simulator_speed)
