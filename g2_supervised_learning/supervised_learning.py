import csv
import random as rnd
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # Used for "mode" - https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
from decision_tree_nodes import DecisionTreeBranchNode, DecisionTreeLeafNode
from matplotlib import lines
from numpy.typing import NDArray
import os # for saving plots
import experiments as ex

# The code below is "starter code" for graded assignment 2 in DTE-2602
# You should implement every method / function which only contains "pass".
# "Helper functions" can be left unedited.
# Feel free to add additional classes / methods / functions to answer the assignment.
# You can use the modules imported above, and you can also import any module included
# in Python 3.10. See https://docs.python.org/3.10/py-modindex.html .
# Using any other modules / external libraries is NOT ALLOWED.


#########################################
#   Data input / prediction evaluation
#########################################


def read_data() -> tuple[NDArray, NDArray]:
    """Read data from CSV file, remove rows with missing data, and normalize

    Returns
    -------
    X: NDArray
        Numpy array, shape (n_samples,4), where n_samples is number of rows
        in the dataset. Contains the four numeric columns in the dataset
        (bill length, bill depth, flipper length, body mass).
        Each column (each feature) is normalized by subtracting the column mean
        and dividing by the column std.dev. ("z-score").
        Rows with missing data ("NA") are discarded.
    y: NDarray
        Numpy array, shape (n_samples,)
        Contains integer values (0, 1 or 2) representing the penguin species

    Notes
    -----
    Z-score normalization: https://en.wikipedia.org/wiki/Standard_score .
    """
    csv_file_path = "palmer_penguins.csv"
    csv_data = []
    with open(csv_file_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for line in csvreader:
            if "NA" in line:
                continue
            csv_data.append(line)
    # header = csv_data[0]
    data = np.array(csv_data[1:])
    X = np.array(data[:,2:6], dtype=float)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X -= mean
    X /= std
    y = data[:,0]
    y[y=="Adelie"] = 0
    y[y=="Chinstrap"] = 1
    y[y=="Gentoo"] = 2
    y = np.array(y, dtype=int)
    return X, y


def convert_y_to_binary(y: NDArray, y_value_true: int) -> NDArray:
    """Convert integer valued y to binary (0 or 1) valued vector

    Parameters
    ----------
    y: NDArray
        Integer valued NumPy vector, shape (n_samples,)
    y_value_true: int
        Value of y which will be converted to 1 in output.
        All other values are converted to 0.

    Returns
    -------
    y_binary: NDArray
        Binary vector, shape (n_samples,)
        1 for values in y that are equal to y_value_true, 0 otherwise
    """
    y_binary = (y==y_value_true).astype(int)
    return y_binary


def train_test_split(
    X: NDArray, y: NDArray, train_frac: float
) -> tuple[tuple[NDArray, NDArray], tuple[NDArray, NDArray]]:
    """Shuffle and split dataset into training and testing datasets

    Parameters
    ----------
    X: NDArray
        Dataset, shape (n_samples,n_features)
    y: NDArray
        Values to be predicted, shape (n_samples)
    train_frac: float
        Fraction of data to be used for training

    Returns
    -------
    (X_train,y_train): tuple[NDArray, NDArray]]
        Training dataset
    (X_test,y_test): tuple[NDArray, NDArray]]
        Test dataset
    """
    ind = np.random.permutation(len(X))
    X = X[ind]
    y = y[ind]
    train_frac *= len(X)
    train_frac = round(train_frac)
    X_train = X[:train_frac]
    y_train = y[:train_frac]
    X_test = X[train_frac:]
    y_test = y[train_frac:]
    return (X_train, y_train), (X_test, y_test)


def accuracy(y_pred: NDArray, y_true: NDArray) -> float:
    """Calculate accuracy of model based on predicted and true values

    Parameters
    ----------
    y_pred: NDArray
        Numpy array with predicted values, shape (n_samples,)
    y_true: NDArray
        Numpy array with true values, shape (n_samples,)

    Returns
    -------
    accuracy: float
        Fraction of cases where the predicted values
        are equal to the true values. Number in range [0,1]

    # Notes:
    See https://en.wikipedia.org/wiki/Accuracy_and_precision#In_classification
    """
    return sum(y_pred==y_true.astype(int)) / len(y_pred)


##############################
#   Gini impurity functions
##############################


def gini_impurity(y: NDArray) -> float:
    """Calculate Gini impurity of a vector

    Parameters
    ----------
    y: NDArray, integers
        1D NumPy array with class labels

    Returns
    -------
    impurity: float
        Gini impurity, scalar in range [0,1)

    # Notes:
    - Wikipedia ref.: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
    """
    unique_values, counts = np.unique(y, return_counts=True)
    impurity = 1
    for count in counts:
        impurity -= (count/len(y))**2
    return impurity


def gini_impurity_reduction(y: NDArray, left_mask: NDArray) -> float:
    """Calculate the reduction in mean impurity from a binary split

    Parameters
    ----------
    y: NDArray
        1D numpy array
    left_mask: NDArray
        1D numpy boolean array, True for "left" elements, False for "right"

    Returns
    -------
    impurity_reduction: float
        Reduction in mean Gini impurity, scalar in range [0,0.5]
        Reduction is measured as _difference_ between Gini impurity for
        the original (not split) dataset, and the _weighted mean impurity_
        for the two split datasets ("left" and "right").

    """
    y_left = y[left_mask]
    y_right = y[~left_mask]
    full_impurity = gini_impurity(y)
    left_impurity = gini_impurity(y_left) * (len(y_left) / len(y))
    right_impurity = gini_impurity(y_right) * (len(y_right) / len(y))
    impurity_reduction = full_impurity - left_impurity - right_impurity
    return impurity_reduction


def best_split_feature_value(X: NDArray, y: NDArray) -> tuple[float, int, float]:
    """Find feature and value "split" that yields highest impurity reduction

    Parameters
    ----------
    X: NDArray
        NumPy feature matrix, shape (n_samples, n_features)
    y: NDArray
        NumPy class label vector, shape (n_samples,)

    Returns
    -------
    impurity_reduction: float
        Reduction in Gini impurity for best split.
        Zero if no split that reduces impurity exists.
    feature_index: int
        Index of X column with best feature for split.
        None if impurity_reduction = 0.
    feature_value: float
        Value of feature in X yielding best split of y
        Dataset is split using X[:,feature_index] <= feature_value
        None if impurity_reduction = 0.

    Notes
    -----
    The method checks every possible combination of feature and
    existing unique feature values in the dataset.
    """
    impurity_reduction = 0
    feature_index = None
    feature_value = None
    for feature in range(X.shape[1]):
        for value in X[:, feature]:
            left_mask = X[:, feature] <= value
            new_reduction = gini_impurity_reduction(y, left_mask)
            if new_reduction > impurity_reduction:
                impurity_reduction = new_reduction
                feature_index = feature
                feature_value = value
    return impurity_reduction, feature_index,  feature_value


###################
#   Perceptron
###################


class Perceptron:
    """Perceptron model for classifying two classes

    Attributes
    ----------
    weights: NDArray
        Array, shape (n_features,), with perceptron weights
    bias: float
        Perceptron bias value
    converged: bool | None
        Boolean indicating if Perceptron has converged during training.
        Set to None if Perceptron has not yet been trained.
    """

    def __init__(self, features=4):
        """Initialize perceptron"""
        self.weights = np.ones(features + 1) # +1 because bias is stored in the weights array
        # self.bias = 0 
        self.converged = None

    def predict_single(self, x: NDArray) -> int:
        """Predict / calculate perceptron output for single observation / row x
        
        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_features,)
        
        Returns
        -------
        y: int
            1 for positive classification, 0 for negative classification.
        """
        x = np.append(x, 1)
        I = self.weights * x
        y = 1 if I >= 0 else 0
        return y


    def predict(self, X: NDArray) -> NDArray:
        """Predict / calculate perceptron output for data matrix X
        
        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        
        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        X = np.append(X, [[1]] * len(X), axis=1)
        I = np.sum(X * self.weights, axis=1)
        y = (I >= 0).astype(int)
        return y

    def train(self, X: NDArray, y: NDArray, learning_rate: float=0.3, max_epochs: int=100):
        """Fit perceptron to training data X with binary labels y

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray 
            NumPy binary class label vector, shape (n_samples,)
        learning_rate: float
            Number in the range [0, 1]
        max_epochs: int
            The maximum number of epochs
        """
        self.converged = False
        epochs = 0
        while not self.converged and epochs < max_epochs:
            old_weights = self.weights.copy()
            for i in range(len(y)):
                features = np.append(X[i, :], 1) # 1 is added so it can be multiplied with bias
                I = sum(self.weights * features)
                V = 1 if I >= 0 else 0
                self.weights = self.weights + learning_rate * (y[i] - V) * features
            epochs += 1
            if np.array_equal(self.weights, old_weights):
                print(f"Converged after {epochs} epoch{"s" if epochs > 1 else ""}")
                self.converged = True
            
        

    def decision_boundary_slope_intercept(self) -> tuple[float, float]:
        """Calculate slope and intercept for decision boundary line (2-feature data only)
        
        Returns
        -------
        slope: float
            Slope of decision boundary line
        intercept: float
            Intercept of decision boundary line
        """
        if len(self.weights) != 3:
            raise ValueError(f"Can not calculate decision boundary line for {len(self.weights)-1}-feature data")
        slope = - self.weights[0] / self.weights[1]
        intercept = - self.weights[2] / self.weights[1]
        return slope, intercept
        


####################
#   Decision tree
####################


class DecisionTree:
    """Decision tree model for classification

    Attributes
    ----------
    _root: DecisionTreeBranchNode | None
        Root node in decision tree
    """

    def __init__(self):
        """Initialize decision tree"""
        self._root = None

    def __str__(self) -> str:
        """Return string representation of decision tree (based on binarytree.Node.__str__())"""
        if self._root is not None:
            return str(self._root)
        else:
            return "<Empty decision tree>"

    def fit(self, X: NDArray, y: NDArray):
        """Train decision tree based on labelled dataset

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray, integers
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        Creates the decision tree by calling _build_tree() and setting
        the root node to the "top" DecisionTreeBranchNode.

        """
        self._root = self._build_tree(X, y)

    def _build_tree(self, X: NDArray, y: NDArray):
        """Recursively build decision tree

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        y: NDArray
            NumPy class label vector, shape (n_samples,)

        Notes
        -----
        - Determines the best possible binary split of the dataset. If no impurity
        reduction can be achieved, a leaf node is created, and its value is set to
        the most common class in y. If a split can achieve impurity reduction,
        a decision (branch) node is created, with left and right subtrees created by
        recursively calling _build_tree on the left and right subsets.

        """
        # Find best binary split of dataset
        impurity_reduction, feature_index, feature_value = best_split_feature_value(
            X, y
        )

        # If impurity can't be reduced further, create and return leaf node
        if impurity_reduction == 0:
            leaf_value = scipy.stats.mode(y, keepdims=False)[0]
            return DecisionTreeLeafNode(leaf_value)

        # If impurity _can_ be reduced, split dataset, build left and right
        # branches, and return branch node.
        else:
            left_mask = X[:, feature_index] <= feature_value
            left = self._build_tree(X[left_mask], y[left_mask])
            right = self._build_tree(X[~left_mask], y[~left_mask])
            return DecisionTreeBranchNode(feature_index, feature_value, left, right)

    def predict(self, X: NDArray):
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)

        Returns
        -------
        y: NDArray, integers
            NumPy class label vector (predicted), shape (n_samples,)
        """
        if self._root is not None:
            return self._predict(X, self._root)
        else:
            raise ValueError("Decision tree root is None (not set)")

    def _predict(
        self, X: NDArray, node: Union["DecisionTreeBranchNode", "DecisionTreeLeafNode"]
    ) -> NDArray:
        """Predict class (y vector) for feature matrix X

        Parameters
        ----------
        X: NDArray
            NumPy feature matrix, shape (n_samples, n_features)
        node: "DecisionTreeBranchNode" or "DecisionTreeLeafNode"
            Node used to process the data. If the node is a leaf node,
            the data is classified with the value of the leaf node.
            If the node is a branch node, the data is split into left
            and right subsets, and classified by recursively calling
            _predict() on the left and right subsets.

        Returns
        -------
        y: NDArray
            NumPy class label vector (predicted), shape (n_samples,)

        Notes
        -----
        The prediction follows the following logic:

            if the node is a leaf node
                return y vector with all values equal to leaf node value
            else (the node is a branch node)
                split the dataset into left and right parts using node question
                predict classes for left and right datasets (using left and right branches)
                "stitch" predictions for left and right datasets into single y vector
                return y vector (length matching number of rows in X)
        """
        if isinstance(node, DecisionTreeLeafNode):
            y = np.full(len(X), node.y_value)
            return y
        left_mask = X[:, node.feature_index] <= node.feature_value
        left = self._predict(X[left_mask, :], node.left)
        right = self._predict(X[~left_mask, :], node.right)
        y = np.zeros(len(X), dtype=int)
        y[left_mask] = left
        y[~left_mask] = right
        return y


############
#   MAIN
############


def run_decision_tree(X, y):
    """Example of using decision tree"""
    train_set, test_set = train_test_split(X, y, 0.7)
    X_train, y_train = train_set
    dTree = DecisionTree()
    dTree.fit(X_train, y_train)
    print(dTree)
    X_test, y_test = test_set
    y_pred = dTree.predict(X_test)
    acc = accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")

def run_perceptron(X, y, y_value_true, features=[True, True, True, True]):
    """Example of using perceptron"""
    X = X[:, features]
    y = convert_y_to_binary(y, y_value_true)
    train_set, test_set = train_test_split(X, y, 0.5)
    X_train, y_train = train_set
    perceptron1 = Perceptron(features=X.shape[1])
    perceptron1.train(X_train, y_train)
    X_test, y_test = test_set
    y_pred = perceptron1.predict(X_test)
    print(f"Is converged: {perceptron1.converged}")
    acc = accuracy(y_pred, y_test)
    print(round(acc * len(y_test)), "/", len(y_test), f"    {round(acc*100, 3)}%")
    print("Perceptron weigths:", perceptron1.weights)
    if X.shape[1] == 2:
        slope, intercept = perceptron1.decision_boundary_slope_intercept()
        draw_plot(X_test, y_test, slope=slope, intercept=intercept)
    else:
        draw_plot(X_test, y_test)

def draw_plot(X, y, y_min=0, y_max=3, fname=None, slope=None, intercept=None):
    """Visualize 2D features X and class label y as scatter plot
    
    Parameters
    ----------
    X: NDArray
        NumPy array with features. Shape (n_samples,2)
    y: NDArray
        NumPy array with integer class labels. Shape (n_samples,)
    y_min: int
        Minimum value for class label (usually zero) 
    y_max: int
        Maximum value for class label (usually n_classes-1)
    fname: str
        File name used to save plot as a file
    slope: float
        Slope from perceptron decision_boundary
    intercept: float
        Intercept from perceptron decision_boundary

    
    Notes
    -----
    y_min and y_max are useful in cases where you want to plot _parts_
    of the dataset which don't contain all class labels. Setting
    y_min and y_max will help keep class colors consistent. If you're 
    plotting the whole dataset, you don't need to set these.
    """
    if len(np.unique(y)) == 3:
        class_label = ["Adelie", "Chinstrap", "Gentoo"]
    else:
        class_label = ["0", "1"]
    for label_value in np.unique(y):
        plt.scatter(x=X[y==label_value, 0],
                    y=X[y==label_value, 1],
                    c=y[y==label_value],
                    label=f'Class {class_label[label_value]}',
                    vmin=y_min, vmax=y_max)
    if slope is not None:
        ax = plt.gca()
        x_line = ax.get_xbound()
        y_line = np.array(x_line) * slope + intercept
        plt.plot(x_line, y_line)
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')
    plt.legend()
    if fname:
        if not os.path.exists("plots"):
            os.makedirs("plots")
        plt.savefig("plots\\" + fname)
    else:
        plt.show()


if __name__ == "__main__":
    # Demonstrate your code / solutions here.
    # Be tidy; don't cut-and-paste lots of lines.
    # Experiments can be implemented as separate functions that are called here.

    X, y = read_data()


    # run_perceptron(X, y, 2, features=[False, False, True, True])
    run_decision_tree(X, y)

    # Experiments:
    # ex.perceptron1(X, y)
    # ex.perceptron2(X, y)
    # ex.decision_tree1(X, y)
    # ex.decision_tree2(X, y)
    # ex.decision_tree3(X, y)

    
