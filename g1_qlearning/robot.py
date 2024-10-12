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
MAP_SIZE = 6

class Robot:
    """A robot that explores unknown terrain."""

    def __init__(self, alpha: float=1.0, gamma: float=0.8) -> None:
        """Initialize the robot

        Keyword arguments:
            alpha -- How much new information is valued (learning rate) (default 1.0)
            gamma -- How much future rewards are valued (discount factor) (default 0.8)
        """
        FLAT = 0
        WATER = -25
        STEEP = -20
        GOAL = 100
        self.OOB_REWARD = -100 # reward for trying to leave the map (Out Of Bounds)
        self.REWARD = [
            [WATER, STEEP, STEEP, FLAT,  FLAT,  WATER],
            [WATER, WATER, FLAT,  STEEP, FLAT,  FLAT],
            [STEEP, FLAT,  FLAT,  STEEP, FLAT,  STEEP],
            [STEEP, FLAT,  FLAT,  FLAT,  FLAT,  FLAT],
            [STEEP, FLAT,  STEEP, FLAT,  STEEP, FLAT],
            [GOAL,  WATER, WATER, WATER, WATER, WATER]
        ]

        self.position = START_POSITION
        self.alpha = alpha # How much new information is valued (learning rate)
        self.gamma = gamma # How much future reward is valued (discount factor)
        self.q_matrix = {}
        for y in range(MAP_SIZE):
            for x in range(MAP_SIZE):
                self.q_matrix[(y, x)] = [0] * 4 # For every square on the map, the robot has 4 choices
        
        
    def get_x(self) -> int:
        """Return the current column of the robot. In the range 0-5."""
        return self.position[1]

    def get_y(self) -> int:
        """Return the current row of the robot. In the range 0-5."""
        return self.position[0]
    
    def get_q_matrix(self) -> dict:
        """Return the robots Q-matrix."""
        return self.q_matrix

    def _get_direction_mc(self) -> int:
        """Return the next direction based on Monte Carlo."""
        direction = random.randint(0, 3)
        return direction

    def _get_direction_eg(self) -> int:
        """Return the next direction based on Epsilon-greedy. If there is a tie, picks randomly from the best options."""
        directions = self.q_matrix[self.position]
        best_directions = [direction for direction in range(4) if directions[direction] == max(directions)]
        direction = random.choice(best_directions)
        return direction
    
    def _get_direction_policy_based(self) -> int:
        """Return direction based on monte carlo or Epsilon-greedy.

        Policy: 50/50 EG/MC
        """
        if random.randint(0, 1) == 0:
            return self._get_direction_mc()
        else:
            return self._get_direction_eg()

    def _get_next_state(self, direction: int) -> tuple[int, int]:
        """Return next state if it is valid, return current state if it is not."""
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
        """Perform simulations and return the highest reward the robot was able to get in a single simulation.
        
        Arguments:
            epochs -- The number of simulations.
        """
        highest_reward = -1000000
        for _ in range(epochs):
            self.position = START_POSITION
            reward = 0
            while not self.has_reached_goal():
                direction = self._get_direction_mc()
                self.position = self._get_next_state(direction)
                y, x = self.position
                reward += self.REWARD[y][x]
            if reward > highest_reward:
                highest_reward = reward
        return highest_reward
    
    def q_learning_converge(self, required_repeats: int=100) -> int:
        """Run q-learning until the Q matrix converges. Return how many epochs it took to converge
        
        Keyword arguments:
            required_repeats -- The number of times the robot has to make it to the goal in a row without changing the Q-matrix.
        """
        epochs = 0
        counter = 0
        while counter < required_repeats:
            epochs += 1
            self.reset_random()
            q_sum = sum(sum(directions) for directions in self.q_matrix.values())
            while not self.has_reached_goal():
                self.one_step_q_learning()
            if q_sum == sum(sum(directions) for directions in self.q_matrix.values()):
                counter += 1
            else: 
                counter = 0
        return epochs - required_repeats


    def q_learning(self, epochs: int) -> None:
        """Run q-learning for a given number of epochs.
        
            Arguments:
                epochs -- The number of simulations of Q-learning.
            """
        for _ in range(epochs):
            self.reset_random()
            while not self.has_reached_goal():
                self.one_step_q_learning()
    
    def one_step_q_learning(self) -> None:
        """Perform one step of q-learning."""
        direction = self._get_direction_policy_based()
        next_position = self._get_next_state(direction)
        y, x = next_position
        # If the robot tries to go out of bounds, next_position will be equal to current position.
        if self.position == next_position: 
            # There is a negative reward for going out of bounds to discourage the robot from trying to leave the map.
            reward = self.OOB_REWARD
        else:
            reward = self.REWARD[y][x]
        self.q_matrix[self.position][direction] = (1 - self.alpha) * self.q_matrix[self.position][direction]\
              + self.alpha * (reward + self.gamma * max(self.q_matrix[next_position]))
        self.position = next_position

    
    def has_reached_goal(self) -> bool:
        """Return True if the robot has reached the goal."""
        return self.position == GOAL_POSITION
        
    def reset_random(self) -> None:
        """Place the robot in a new random state."""
        self.position = (random.randint(0, 5), random.randint(0, 5))

    def greedy_path(self) -> tuple[list, int]:
        """Use epsilon-greedy on the values in the q-matrix to create a path from the start to the goal.
        
        Return the path and reward the robot was able to collect. 
        The function stops adding to the path if the length goes over 100
        """
        self.position = START_POSITION
        path = [START_POSITION]
        reward = 0
        while not self.has_reached_goal() and len(path) < 100:
            direction = self._get_direction_eg()
            self.position = self._get_next_state(direction)
            path.append(self.position)
            y, x = self.position
            reward += self.REWARD[y][x]
        return path, reward

# Feel free to add additional classes / methods / functions to solve the assignment...