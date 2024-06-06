import pandas as pd
import numpy as np

class Node:
    def __init__(self, attribute=None, value=None, decision=None):
        self.attribute = attribute
        self.value = value
        self.decision = decision
        self.children = {}

def entropy(labels):
    n = len(labels)
    if n <= 1:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / n
    return -np.sum(probabilities * np.log2(probabilities))

def information_gain(data, attribute, target_attribute):
    total_entropy = entropy(data[target_attribute])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[attribute] == val).dropna()[target_attribute]) for i, val in enumerate(values)])
    return total_entropy - weighted_entropy

def build_tree(data, attributes, target_attribute):
    if len(np.unique(data[target_attribute])) == 1:
        return Node(decision=data[target_attribute].iloc[0])

    if len(attributes) == 0:
        return Node(decision=data[target_attribute].mode().iloc[0])

    information_gains = [information_gain(data, attribute, target_attribute) for attribute in attributes]
    best_attribute_index = np.argmax(information_gains)
    best_attribute = attributes[best_attribute_index]

    tree = Node(attribute=best_attribute)

    for value in np.unique(data[best_attribute]):
        sub_data = data.where(data[best_attribute] == value).dropna()
        subtree = build_tree(sub_data, [attr for attr in attributes if attr != best_attribute], target_attribute)
        tree.children[value] = subtree

    return tree

def predict(node, instance):
    if node.decision is not None:
        return node.decision
    else:
        return predict(node.children[instance[node.attribute]], instance)

# Define the dataset
data = {
    'day': list(range(1, 15)),
    'outlook': ['sunny', 'sunny', 'overcast', 'rain', 'rain', 'rain', 'overcast', 'sunny', 'sunny', 'rain', 'sunny', 'overcast', 'overcast', 'rain'],
    'temperature': ['hot', 'hot', 'hot', 'mild', 'cool', 'cool', 'cool', 'mild', 'cool', 'mild', 'mild', 'mild', 'hot', 'mild'],
    'humidity': ['high', 'high', 'high', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'normal', 'normal', 'high', 'normal', 'high'],
    'wind': ['weak', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'weak', 'weak', 'strong', 'strong', 'weak', 'strong'],
    'playing': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define attributes and target attribute
attributes = df.columns[1:-1]
target_attribute = df.columns[-1]

# Build the decision tree
tree = build_tree(df, attributes, target_attribute)

# Test the decision tree with a sample instance
sample_instance = {'outlook': 'rain', 'temperature': 'mild', 'humidity': 'normal', 'wind': 'weak'}
prediction = predict(tree, sample_instance)
print("Sample Instance Decision:", prediction)
