from decision_trees import tree_plotter
from decision_trees import trees


FILE_NAME = "decision_trees/lenses.txt"
lenses = [line.strip().split("\t") for line in open(FILE_NAME, "r").readlines()]
labels = ["age", "prescript", "astigmatic", "tearRate"]
lensees_tree = trees.create_tree(lenses, labels)

tree_plotter.create_plot(lensees_tree)