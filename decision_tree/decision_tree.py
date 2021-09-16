import numpy as np
import pandas as pd
# IMPORTANT: DO NOT USE ANY OTHER 3RD PARTY PACKAGES
# (math, random, collections, functools, etc. are perfectly fine)


class DecisionTree:

    def __init__(self):
        # NOTE: Feel free add any hyperparameters
        # (with defaults) as you see fit
        self.tree = None
        pass

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Generates a decision tree for classification

        Args:
            X (pd.DataFrame): a matrix with discrete value where
                each row is a sample and the columns correspond
                to the features.
            y (pd.Series): a vector of discrete ground-truth labels
        """
        # TODO: Implement

        root = Node()
        if (y == 'Yes').all():
            root.label = 'Yes'
            root.leaf = True
            return root
        elif (y == 'No').all():
            root.label = 'No'
            root.leaf = True
            return root
        elif X.empty:
            root.label = y.mode()[0]  # Most common label
            root.leaf = True
            return root
        else:
            A = getBestAttribute(X, y)
            root.decision_attribute = A
            for v_i in X[A].unique():  # For each possible value, vi, of A
                new_branch = Node(parent=root, label=v_i)
                root.addChild(new_branch)
                X_vi = X.loc[X[A] == v_i]
                y_vi = y.loc[X[A] == v_i]
                if X_vi.empty:
                    new_branch.addChild(
                        Node(parent=new_branch, label=y.mode()[0], leaf=True))
                else:
                    new_branch.addChild(self.fit(X_vi, y_vi))
        # The first call will be the last to return, so this will be the top-level root
        self.tree = root
        return root

    def predict(self, X: pd.DataFrame):
        """
        Generates predictions

        Note: should be called after .fit()

        Args:
            X (pd.DataFrame): an mxn discrete matrix where
                each row is a sample and the columns correspond
                to the features.

        Returns:
            A length m vector with predictions
        """
        # TODO: Implement
        predictions = []
        for index, row in X.iterrows():
            predictions.append(self.predictSingle(row))
        return pd.Series(predictions)

        raise NotImplementedError()

    def predictSingle(self, singleRow: pd.Series):
        node = self.tree
        while node.children:
            for child in node.children:
                if child.leaf:
                    return child.label
                if child.decision_attribute is not None:
                    node = child
                    break
                if child.label == singleRow[node.decision_attribute]:
                    node = child
                    break
        return node.label

    def get_rules(self):
        """
        Returns the decision tree as a list of rules

        Each rule is given as an implication "x => y" where
        the antecedent is given by a conjuction of attribute
        values and the consequent is the predicted label

            attr1=val1 ^ attr2=val2 ^ ... => label

        Example output:
        >>> model.get_rules()
        [
            ([('Outlook', 'Overcast')], 'Yes'),
            ([('Outlook', 'Rain'), ('Wind', 'Strong')], 'No'),
            ...
        ]
        """
        # TODO: Implement
        raise NotImplementedError()


def getBestAttribute(X, y) -> str:
    # TODO: Implement
    attr = X.columns[np.argmax(
        [entropy(y.groupby(X[col]).count()) for col in X.columns])]
    return attr

    # raise NotImplementedError()


class Node:
    def __init__(self, parent=None, label=None, leaf=False):
        self.parent = parent
        self.decision_attribute = None
        self.label = label  # Value of decision attribute of parent node
        self.children = []
        self.leaf = leaf
        pass

    def addChild(self, child):
        self.children.append(child)

# --- Some utility functions


def accuracy(y_true, y_pred):
    """
    Computes discrete classification accuracy

    Args:
        y_true (array<m>): a length m vector of ground truth labels
        y_pred (array<m>): a length m vector of predicted labels

    Returns:
        The average number of correct predictions
    """
    assert y_true.shape == y_pred.shape
    return (y_true == y_pred).mean()


def entropy(counts):
    """
    Computes the entropy of a partitioning

    Args:
        counts (array<k>): a lenth k int array >= 0. For instance,
            an array [3, 4, 1] implies that you have a total of 8
            datapoints where 3 are in the first group, 4 in the second,
            and 1 one in the last. This will result in entropy > 0.
            In contrast, a perfect partitioning like [8, 0, 0] will
            result in a (minimal) entropy of 0.0

    Returns:
        A positive float scalar corresponding to the (log2) entropy
        of the partitioning.

    """
    assert (counts >= 0).all()
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Avoid log(0)
    return - np.sum(probs * np.log2(probs))


if __name__ == '__main__':
    data_1 = pd.read_csv(
        'C:/Users/Håvard/Github/machine_learning/task-1/decision_tree/data_1.csv')
    # Separate independent (X) and dependent (y) variables
    X = data_1.drop(columns=['Play Tennis'])
    y = data_1['Play Tennis']

    # Create and fit a Decrision Tree classifier
    model_1 = DecisionTree()  # <-- Should work with default constructor
    model_1.fit(X, y)
    model_1.predict(X)
