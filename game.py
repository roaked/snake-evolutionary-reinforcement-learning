import random, pygame, sys, time
from enum import Enum
from collections import namedtuple
import numpy as np

check_errors = pygame.init() 
if check_errors[1] > 0:
    print(f'[!] Had {check_errors[1]} errors when initialising game, exiting...')
    sys.exit(-1)
else:
    print('[+] Game successfully initialised')

font = pygame.font.SysFont('ContrailOne-Regular.ttf', 50) # add font

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')


# DIFFICULTY
# Easy      -> 10
# Medium    ->  25
# Hard      ->  40
# Harder    ->  60
# Impossible->  120
DIFFICULTY = 25

# WINDOW SIZE
HEIGHT = 680
WIDTH = 680

# RGB COLORS
CYAN = (25,140,140)
DARKCYAN = (0,102,102)
WHITE = (255, 255, 255)
RED = (200,0,0)
GREY = (22,22,22)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# SETTINGS
BLOCK_SIZE = 20
SPEED = 20

class SnakeGameAI:

    def __init__(self, width = WIDTH, height = HEIGHT):
        #window display
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self._init_game()


    def _init_game(self):
        # Initializing game state variables (snake position, score, food)
        start_x = self.width // 2
        start_y = self.height // 2

        self.direction = Direction.RIGHT
        self.head = Point(start_x, start_y)
        self.snake = [self.head]

        # Create initial instance for the snake with three blocks
        for i in range(1, 3):
            self.snake.append(Point(self.head.x - i * BLOCK_SIZE, start_y))

        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()
        self.deaths = 0 # need to update this later
        self.steps = 0 # update this in play step


    def _place_food(self):
        snake_positions = {point for point in self.snake}  # Set of snake positions

        # Generate random positions until an available one is found
        while True:
            x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE

            new_food_position = Point(x, y)
            if new_food_position not in snake_positions:
                self.food = new_food_position
                break


    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head)
        self.steps += 1

        # Check for game over conditions
        game_over = False
        reward = 0
        collision_or_frame_limit = self.is_collision() or self.frame_iteration > 100 * len(self.snake)
        if collision_or_frame_limit:
            game_over = True
            reward = -10
            self.deaths += 1 # dying
            self.steps = 0
            return reward, game_over, self.score, self.deaths, self.steps
        
        # Check if the snake has eaten the food
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
            self.steps = 0
        else:
            self.snake.pop()
        
        # Update UI and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # Return game status and score
        return reward, game_over, self.score, self.deaths, self.steps


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head

        # Boundary collision check
        collides_with_boundary = (
            pt.x > self.width - BLOCK_SIZE or
            pt.x < 0 or
            pt.y > self.height - BLOCK_SIZE or
            pt.y < 0
        )

        # Snake body collision check
        collides_with_snake = pt in self.snake[1:]
        
        return collides_with_boundary or collides_with_snake


    def _update_ui(self):
        self.display.fill(GREY)

        # Draw the snake
        for pt in self.snake:
            snake_rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.display, BLUE1, snake_rect)
            pygame.draw.rect(self.display, BLUE2, snake_rect.inflate(-8, -8))  # Inflate the rectangle

        # Draw the food
        food_rect = pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.display, RED, food_rect)

        # Render the score text
        score_text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(score_text, (0, 0))

        # Update the display
        pygame.display.flip()


    def _move(self, action):
        # Define clockwise directions
        clockwise_directions = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        current_direction = self.direction

        # Define action codes for each movement direction
        no_change_action = [1, 0, 0]
        right_turn_action = [0, 1, 0]
        left_turn_action = [0, 0, 1]

        # Determine the change in direction based on action
        if action == no_change_action:
            direction_change = 0
        elif action == right_turn_action:
            direction_change = 1
        elif action == left_turn_action:
            direction_change = -1

        # Calculate the new direction index
        current_index = clockwise_directions.index(current_direction)
        new_index = (current_index + direction_change) % 4
        new_direction = clockwise_directions[new_index]

        # Update the direction
        self.direction = new_direction

        # Define movement adjustments for each direction
        movement_adjustments = {
            Direction.RIGHT: (BLOCK_SIZE, 0),
            Direction.LEFT: (-BLOCK_SIZE, 0),
            Direction.DOWN: (0, BLOCK_SIZE),
            Direction.UP: (0, -BLOCK_SIZE)
        }

        # Apply movement adjustments to update the head position
        move_x, move_y = movement_adjustments[self.direction]
        self.head = Point(self.head.x + move_x, self.head.y + move_y)