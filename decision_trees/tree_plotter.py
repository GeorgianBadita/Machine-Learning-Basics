import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")

decision_node = {"boxstyle": "sawtooth", "fc": "0.8"}
leaf_node = {"boxstyle": "round4", "fc": "0.8"}
arrow_args = {"arrowstyle": "<-"}


def plot_node(node_text, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(
        node_text,
        xy=parent_pt,
        xycoords="axes fraction",
        xytext=center_pt,
        textcoords="axes fraction",
        va="center",
        ha="center",
        bbox=node_type,
        arrowprops=arrow_args,
    )


def plot_mid_text(cnt_pt, parent_pt, txt_string):
    x_mid = (parent_pt[0] - cnt_pt[0]) / 2.0 + cnt_pt[0]
    y_mid = (parent_pt[1] - cnt_pt[1]) / 2.0 + cnt_pt[1]
    create_plot.ax1.text(
        x_mid, y_mid, txt_string, va="center", ha="center", rotation=30
    )


def plot_tree(
    tree, parent_pt, node_txt
):  # if the first key tells you what feat was split on
    numLeafs = get_num_leafs(tree)  # this determines the x width of this tree
    firstStr = list(tree.keys())[0]  # the text label for this node should be this
    cnt_pt = (
        plot_tree.xOff + (1.0 + float(numLeafs)) / 2.0 / plot_tree.totalW,
        plot_tree.yOff,
    )
    plot_mid_text(cnt_pt, parent_pt, node_txt)
    plot_node(firstStr, cnt_pt, parent_pt, decision_node)
    secondDict = tree[firstStr]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in secondDict.keys():
        if (
            type(secondDict[key]).__name__ == "dict"
        ):  # test to see if the nodes are dictonaires, if not they are leaf nodes
            plot_tree(secondDict[key], cnt_pt, str(key))  # recursion
        else:  # it's a leaf node print the leaf node
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(
                secondDict[key], (plot_tree.xOff, plot_tree.yOff), cnt_pt, leaf_node
            )
            plot_mid_text((plot_tree.xOff, plot_tree.yOff), cnt_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


# if you do get a dictonary you know it's a tree, and the first element will be another dict


def create_plot(tree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)  # no ticks
    # createPlot.ax1 = plt.subplot(111, frameon=False) #ticks for demo puropses
    plot_tree.totalW = float(get_num_leafs(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree, (0.5, 1.0), "")
    plt.show()


def get_num_leafs(tree):
    num_leafs = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]

    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(tree):
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == "dict":
            depth = 1 + get_tree_depth(second_dict[key])
        else:
            depth = 1
    max_depth = max(max_depth, depth)
    return max_depth


def retrieve_tree(i):
    list_of_trees = [
        {"no surfacing": {0: "no", 1: {"flippers": {0: "no", 1: "yes"}}}},
        {
            "no surfacing": {
                0: "no",
                1: {"flippers": {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}},
            }
        },
    ]

    return list_of_trees[i]
