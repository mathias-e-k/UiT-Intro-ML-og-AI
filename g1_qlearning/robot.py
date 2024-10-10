# You can import matplotlib or numpy, if needed.
# You can also import any module included in Python 3.10, for example "random".
# See https://docs.python.org/3.10/py-modindex.html for included modules.
import random

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
START_POSITION = (0, 3)
GOAL_POSITION = (5, 0)

class Robot:
    """a robot!"""
    def __init__(self) -> None:
        FLAT = 0
        WATER = -20
        STEEP = -15
        GOAL = 100
        self.OOB_REWARD = -100
        self.REWARD = [
            [WATER, STEEP, STEEP, FLAT,  FLAT,  WATER],
            [WATER, WATER, FLAT,  STEEP, FLAT,  FLAT],
            [STEEP, FLAT,  FLAT,  STEEP, FLAT,  STEEP],
            [STEEP, FLAT,  FLAT,  FLAT,  FLAT,  FLAT],
            [STEEP, FLAT,  STEEP, FLAT,  STEEP, FLAT],
            [GOAL,  WATER, WATER, WATER, WATER, WATER]
        ]
        self.position = START_POSITION
        self.alpha = 1
        self.gamma = 0.8
        self.q_matrix = {}
        for y in range(6):
            for x in range(6):
                self.q_matrix[(y, x)] = [0, 0, 0, 0]
        
        
    def get_x(self) -> int:
        """Returns the current column of the robot. In the range 0-5"""
        return self.position[1]

    def get_y(self) -> int:
        """Returns the current row of the robot. In the range 0-5"""
        return self.position[0]
    
    def get_q_matrix(self) -> dict:
        return self.q_matrix

    def get_direction_mc(self):
        """Returns the next direction based on Monte Carlo."""
        direction = random.randint(0, 3)
        return direction

    def get_direction_eg(self):
        """Returns the next direction based on Epsilon-greedy. If there is a tie, picks randomly from the best options"""
        directions = self.q_matrix[self.position]
        max_directions = [i for i in range(4) if directions[i] == max(directions)]
        direction = random.choice(max_directions)
        return direction

    def get_next_state(self, direction: int) -> tuple[int, int]:
        """Returns next state if it is valid, otherwise returns current state."""
        if direction == UP and self.position[0] - 1 >= 0:
            return (self.position[0] - 1, self.position[1])
        
        if direction == DOWN and self.position[0] + 1 <= 5:
            return (self.position[0] + 1, self.position[1])
        
        if direction == LEFT and self.position[1] - 1 >= 0:
            return (self.position[0], self.position[1] - 1)
        
        if direction == RIGHT and self.position[1] + 1 <= 5:
            return (self.position[0], self.position[1] + 1)
        
        return self.position
        

    def monte_carlo_exploration(self, epochs: int) -> int:
        """Performs n simulations and returns the highest reward the robot was able to get in a single simulation"""
        highest_reward = -1000000
        for _ in range(epochs):
            self.position = (0, 3)
            reward = 0
            while not self.has_reached_goal():
                direction = self.get_direction_mc()
                self.position = self.get_next_state(direction)
                y, x = self.position
                reward += self.REWARD[y][x]
            if reward > highest_reward:
                highest_reward = reward
        return highest_reward

    def get_direction_policy_based(self) -> int:
        """Returns direction based on monte carlo or Epsilon-greedy
        Policy: 50/50 EG/MC
        """
        if random.randint(0, 1) == 0:
            return self.get_direction_eg()
        else:
            return self.get_direction_mc()
    
    def q_learning_converge(self) -> int:
        """Runs q-learning until the Q matrix converges"""
        epochs = 0
        counter = 0
        while counter < 10:
            epochs += 1
            self.reset_random()
            q_sum = sum(sum(value) for value in self.q_matrix.values())
            while not self.has_reached_goal():
                self.one_step_q_learning()
            if q_sum == sum(sum(value) for value in self.q_matrix.values()):
                counter += 1
            else: 
                counter = 0
        return epochs - 10



    def q_learning(self, epochs: int) -> None:
        """Runs q-learning for a given number of epochs"""
        for _ in range(epochs):
            self.reset_random()
            while not self.has_reached_goal():
                self.one_step_q_learning()
    
    def one_step_q_learning(self):
        """Performs one step of q-learning"""
        direction = self.get_direction_policy_based()
        next_position = self.get_next_state(direction)
        y, x = next_position
        if self.position == next_position:
            reward = self.OOB_REWARD
        else:
            reward = self.REWARD[y][x]
        self.q_matrix[self.position][direction] = (1 - self.alpha) * self.q_matrix[self.position][direction]\
              + self.alpha * (reward + self.gamma * max(self.q_matrix[next_position]))
        self.position = next_position

    
    def has_reached_goal(self) -> bool:
        """Returns True if the robot has reached the goal"""
        return self.position == GOAL_POSITION
        
    def reset_random(self):
        """Place the robot in a new random state."""
        self.position = (random.randint(0, 5), random.randint(0, 5))

    def greedy_path(self):
        self.position = START_POSITION
        path = [START_POSITION]
        reward = 0
        i = 0
        while not self.has_reached_goal() and i < 100:
            direction = self.get_direction_eg()
            self.position = self.get_next_state(direction)
            path.append(self.position)
            y, x = self.position
            reward += self.REWARD[y][x]
            i += 1
        return path, reward

# Feel free to add additional classes / methods / functions to solve the assignment...