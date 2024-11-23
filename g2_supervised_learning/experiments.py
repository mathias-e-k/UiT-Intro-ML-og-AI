import supervised_learning as sl

def perceptron1(X, y):
    X = X[:, [False, True, True, False]]
    y = sl.convert_y_to_binary(y, 2)
    train_set, test_set = sl.train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    perceptron1 = sl.Perceptron(features=X.shape[1])
    perceptron1.train(X_train, y_train)
    X_test, y_test = test_set
    y_pred = perceptron1.predict(X_test)
    print(f"Is converged: {perceptron1.converged}")
    acc = sl.accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")
    print("Perceptron weigths:", perceptron1.weights)
    slope, intercept = perceptron1.decision_boundary_slope_intercept()
    sl.draw_plot(X_train, y_train, slope=slope, intercept=intercept)

def perceptron2(X, y):
    X = X[:, [True, True, False, False]]
    y = sl.convert_y_to_binary(y, 1)
    train_set, test_set = sl.train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    perceptron1 = sl.Perceptron(features=X.shape[1])
    perceptron1.train(X_train, y_train)
    X_test, y_test = test_set
    y_pred = perceptron1.predict(X_test)
    print(f"Is converged: {perceptron1.converged}")
    acc = sl.accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")
    print("Perceptron weigths:", perceptron1.weights)
    slope, intercept = perceptron1.decision_boundary_slope_intercept()
    sl.draw_plot(X_train, y_train, slope=slope, intercept=intercept)

def decision_tree1(X, y):
    X = X[:, [False, True, True, False]]
    y = sl.convert_y_to_binary(y, 2)
    train_set, test_set = sl.train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    dTree = sl.DecisionTree()
    dTree.fit(X_train, y_train)
    print(dTree)
    X_test, y_test = test_set
    y_pred = dTree.predict(X_test)
    acc = sl.accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")

def decision_tree2(X, y):
    X = X[:, [True, True, False, False]]
    y = sl.convert_y_to_binary(y, 1)
    train_set, test_set = sl.train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    dTree = sl.DecisionTree()
    dTree.fit(X_train, y_train)
    print(dTree)
    X_test, y_test = test_set
    y_pred = dTree.predict(X_test)
    acc = sl.accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")

def decision_tree3_once(X, y):
    train_set, test_set = sl.train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    dTree = sl.DecisionTree()
    dTree.fit(X_train, y_train)
    X_test, y_test = test_set
    y_pred = dTree.predict(X_test)
    acc = sl.accuracy(y_pred, y_test)
    return acc

def decision_tree3(X, y, epochs=100):
    sum = 0
    for _ in range(epochs):
        sum += decision_tree3_once(X, y)
    average_accurace = sum / epochs
    print("average accurace:", average_accurace)


if __name__ == "__main__":
    X, y = sl.read_data()
    decision_tree3(X, y)