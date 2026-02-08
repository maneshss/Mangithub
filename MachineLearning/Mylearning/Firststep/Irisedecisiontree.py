import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz

iris = load_iris()

print("data", iris.data[:5])
print("Target", iris.target[:5])

X = iris.data[:, 2:]  # petal length and width
print("X shape:", X.shape)
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_clf.fit(X, y)

# ---------- GINI CALCULATION (based on the trained tree) ----------
t = tree_clf.tree_
root = 0


def gini_node(node_id):
    """Return (gini, counts) for a node in a fitted sklearn DecisionTree."""
    counts = t.value[node_id][0]  # class counts at node
    total = counts.sum()
    p = counts / total
    g = 1.0 - np.sum(p**2)
    return g, counts


# Root and its children
left = t.children_left[root]
right = t.children_right[root]

g_root, c_root = gini_node(root)
g_left, c_left = gini_node(left)
g_right, c_right = gini_node(right)

# Weighted gini after the root split + gini gain
n_left = c_left.sum()
n_right = c_right.sum()
g_weighted = (n_left / (n_left + n_right)) * g_left + (
    n_right / (n_left + n_right)
) * g_right
g_gain = g_root - g_weighted

# Root split info
feat_idx = t.feature[root]
thr = t.threshold[root]
feat_name = iris.feature_names[2:][feat_idx] if feat_idx != -2 else "leaf"

print("\n=== Gini impurity (root split) ===")
print(f"Root split: {feat_name} <= {thr:.4f}")
print("Root counts:", c_root.astype(int), "Gini:", round(g_root, 4))
print("Left counts:", c_left.astype(int), "Gini:", round(g_left, 4))
print("Right counts:", c_right.astype(int), "Gini:", round(g_right, 4))
print("Weighted Gini after split:", round(g_weighted, 4))
print("Gini decrease (gain):", round(g_gain, 4))

# ---------- EXPORT TREE ----------
export_graphviz(
    tree_clf,
    out_file="iris_tree.dot",
    feature_names=iris.feature_names[2:],
    class_names=iris.target_names,
    rounded=True,
    filled=True,
)

# ---------- RENDER PNG ----------
from graphviz import Source

Source.from_file("iris_tree.dot").render("iris_tree", format="png", cleanup=True)
print("\nSaved iris_tree.png")
