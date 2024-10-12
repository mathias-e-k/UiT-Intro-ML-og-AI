import time
start_time = time.time()
# print("--- %s seconds ---" % (time.time() - start_time))
import robot
sum = 0

rounds = 200


for i in range(rounds):
    r1 = robot.Robot(alpha=1)
    sum += r1.q_learning_converge()
    print(i)
print(sum / rounds)
print("--- %s seconds ---" % ((time.time() - start_time) / rounds))

