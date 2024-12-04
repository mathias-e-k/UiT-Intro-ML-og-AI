import random
graph_adjacency_list = {
    "A" : ["B", "C"],
    "B" : ["E", "F"],
    "C" : ["D", "E", "F"],
    "D" : ["F", "G"],
    "E" : None,
    "F" : None,
    "G" : None
    }
Results = {"E" : 0, "F" : 0, "G" : 0}

def simulate(epochs):
    for _ in range(epochs):
        node = "A"
        while node not in "EFG":
            node = random.choice(graph_adjacency_list[node])
        Results[node] += 1

EPOCHS = 1000000
simulate(EPOCHS)
for k in Results:
    print(k, f"{100 * Results[k] / EPOCHS}%")
