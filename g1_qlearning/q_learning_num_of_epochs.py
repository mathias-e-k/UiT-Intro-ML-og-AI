EPOCHS = 9 # Number of epochs per round
ROUNDS = 10000 # Increase the number of rounds to get more accurate results. Takes more time
sum = 0
from robot import Robot
for i in range(ROUNDS):
    r1 = Robot(alpha=1.0)
    r1.q_learning(epochs=EPOCHS)
    path, reward = r1.greedy_path()
    sum += reward
print(f"On average the robots reward for its best path after {EPOCHS} epochs was {sum / ROUNDS}")
