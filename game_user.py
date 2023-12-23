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
GREY = (18,18,18)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

# SETTINGS
BLOCK_SIZE = 20
SPEED2 = 10

class SnakeGameUser:
    
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
        self.steps = 0 # update this in play steps
        
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
        
    def play_step(self):
        self.frame_iteration +=1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = Direction.DOWN
        
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        self.steps += 1
        
        game_over = False # check for game over
        reward = 0
        collision_or_frame_limit = self.is_collision() or self.frame_iteration > 100 * len(self.snake)
        if collision_or_frame_limit:
            game_over = True
            reward = -10
            self.deaths += 1 # dying
            self.steps = 0
            return reward, game_over, self.score, self.deaths, self.steps
        
        if self.head == self.food: # spawn new food --> eat current food
            self.score += 1
            for blocks in range(10):  
                self.snake.append(Point(self.food.x, self.food.y))
            self._place_food()
            self.steps = 0
        else:
            self.snake.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED2)

        return game_over, self.score, self.steps, self.deaths
    
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
        self.display.fill(BLACK)

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
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

if __name__ == "__main__":   
    
    game = SnakeGameUser()            
    # game loop
    while True:
        game_over, score, steps, deaths = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
        
    pygame.quit()