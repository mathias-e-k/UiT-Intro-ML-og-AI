import supervised_learning as sl
import matplotlib.pyplot as plt
import numpy as np
def perceptron1(X, y):
    X = X[:, [False, True, True, False]]
    y = sl.convert_y_to_binary(y, 2)
    train_set, test_set = sl.train_test_split(X, y, 0.4)
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
    train_set, test_set = sl.train_test_split(X, y, 0.4)
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
    train_set, test_set = sl.train_test_split(X, y, 0.4)
    X_train, y_train = train_set
    dTree = sl.DecisionTree()
    dTree.fit(X_train, y_train)
    print(dTree)
    X_test, y_test = test_set
    y_pred = dTree.predict(X_test)
    acc = sl.accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")
    # sl.draw_plot(X_test, y_test)

def decision_tree2(X, y):
    X = X[:, [True, True, False, False]]
    y = sl.convert_y_to_binary(y, 1)
    train_set, test_set = sl.train_test_split(X, y, 0.4)
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
    correct = round(acc * len(y_test))
    return acc, correct

def decision_tree3(X, y, epochs=100):
    sum = 0
    totals = []
    for i in range(epochs):
        print(i, "/", epochs)
        acc, correct = decision_tree3_once(X, y)
        sum += acc
        totals.append(correct)
    average_accuracy = sum / epochs
    results = np.array(totals)
    avg = np.average(results)
    std = np.std(results)
    print("average accuracy:", round(average_accuracy, 4), "avg", avg, "std:", std)
    print("worst:", min(totals), "Best:", max(totals))

def perceptron_train_set_percentage(X, y):
    """Calculate the average accuracy for every train set percentage."""
    EPOCHS = 100
    y = sl.convert_y_to_binary(y, 2)
    sums = [0] * 334
    for i in range(334):
        print(f"{i}/333")
        for _ in range(EPOCHS):
            train_set, test_set = sl.train_test_split(X, y, i/334)
            X_train, y_train = train_set
            perceptron1 = sl.Perceptron()
            perceptron1.train(X_train, y_train)
            X_test, y_test = test_set
            y_pred = perceptron1.predict(X_test)
            acc = sl.accuracy(y_pred, y_test)
            sums[i] += acc
        sums[i] /= EPOCHS
    x_line = [i/334 for i in range(334)]
    plt.plot(x_line, sums)
    plt.xlabel('train set %')
    plt.ylabel('Accuracy')
    plt.savefig("plots\\p_accuracy2")
    plt.show()

def decision_tree_train_set_percentage(X, y):
    """Calculate the average accuracy for every train set percentage."""
    EPOCHS = 3
    sums = [0] * 334
    for i in range(334):
        print(f"{i}/333")
        for _ in range(EPOCHS):
            train_set, test_set = sl.train_test_split(X, y, i/334)
            X_train, y_train = train_set
            dTree = sl.DecisionTree()
            dTree.fit(X_train, y_train)
            X_test, y_test = test_set
            y_pred = dTree.predict(X_test)
            acc = sl.accuracy(y_pred, y_test)
            sums[i] += acc
        sums[i] /= EPOCHS
    x_line = [i/334 for i in range(334)]
    plt.plot(x_line[15:], sums[15:])
    plt.xlabel('train set %')
    plt.ylabel('Accuracy')
    plt.savefig("plots\\dt_accuracy_15+")
    plt.show()

def perceptron_once(X, y, y_value_true, features=[True, True, True, True]):
    """Example of using perceptron"""
    X = X[:, features]
    y = sl.convert_y_to_binary(y, y_value_true)
    train_set, test_set = sl.train_test_split(X, y, 0.4)
    X_train, y_train = train_set
    perceptron1 = sl.Perceptron(features=X.shape[1])
    perceptron1.train(X_train, y_train)
    X_test, y_test = test_set
    y_pred = perceptron1.predict(X_test)
    acc = sl.accuracy(y_pred, y_test)
    correct = round(acc * len(y_test))
    return acc, correct
    

def perceptron_stats(X, y, epochs=100):
    sum = 0
    totals = []
    for i in range(epochs):
        print(i, "/", epochs)
        acc, correct = perceptron_once(X, y, 1, features=[True, True, False, False])
        sum += acc
        totals.append(correct)
    average_accuracy = sum / epochs
    results = np.array(totals)
    avg = np.average(results)
    std = np.std(results)
    print("average accuracy:", round(average_accuracy, 4), "avg", avg, "std:", std)
    print("worst:", min(totals), "Best:", max(totals))

if __name__ == "__main__":
    X, y = sl.read_data()
    # decision_tree_train_set_percentage(X, y)
    # perceptron_train_set_percentage(X, y)
    # perceptron_stats(X, y)
    decision_tree3(X, y)