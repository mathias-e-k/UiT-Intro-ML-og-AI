import robot

r1 = robot.Robot()
print(r1.q_learning_converge())
q = r1.get_q_matrix()
for key in q:
    print(key, q[key])
print(r1.greedy_path())