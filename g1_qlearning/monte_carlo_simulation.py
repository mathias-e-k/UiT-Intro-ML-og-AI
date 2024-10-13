SIMULATIONS = 1000 # Number of simulations per round
ROUNDS = 10 # Increase the number of rounds to get more accurate results. Takes more time
sum = 0
from robot import Robot
r1 = Robot()
for i in range(ROUNDS):
    sum += r1.monte_carlo_exploration(SIMULATIONS)
print(f"On average the robots best score after {SIMULATIONS} simulations was {sum / ROUNDS}")
