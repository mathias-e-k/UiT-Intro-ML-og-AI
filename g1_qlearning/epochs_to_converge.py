ALPHA = 1.0 # learning rate
ROUNDS = 30 # Increase the number of rounds to get more accurate results. Takes more time
sum = 0
from robot import Robot
for i in range(ROUNDS):
    r1 = Robot(alpha=ALPHA)
    sum += r1.q_learning_converge()
print(f"On average the Q-matrix converged after {sum / ROUNDS} epochs when the alpha was {ALPHA}")
