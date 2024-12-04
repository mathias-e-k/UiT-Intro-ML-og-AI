import numpy as np 

X = np.array([[57, 155], [71, 177], [78, 183], [69, 170], [75, 179], [66, 165]])
y = np.array(["F", "M", "M", "F", "M", "F"])

avg = np.average(X[y=="M", 1])
print(avg)