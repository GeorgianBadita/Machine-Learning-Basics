import operator
from math import log


def create_data_set():
    data_set = [[1, 1, "yes"], [1, 1, "yes"], [1, 0, "no"], [0, 1, "no"], [0, 1, "no"]]
    labels = ["no surfacing", "flippers"]
    return data_set, labels


def calc_shannon_ent(data_set):
    num_entires = len(data_set)

    label_counts = {}
    for feat_vec in data_set:
        label = feat_vec[-1]
        if label not in label_counts:
            label_counts[label] = 0
        label_counts[label] += 1

    ent = 0
    for key in label_counts:
        prob = label_counts[key] / num_entires
        ent -= prob * log(prob, 2)
    return ent


def split_data_set(data_set, axis, value):
    ret_data_set = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            reduced_feat_vec = feat_vec[:axis]
            reduced_feat_vec.extend(feat_vec[axis + 1 :])
            ret_data_set.append(reduced_feat_vec)

    return ret_data_set


def choose_best_feature_to_split(data_set):
    num_features = len(data_set[0]) - 1
    base_ent = calc_shannon_ent(data_set)

    best_info_gain = 0.0
    best_feature = -1

    for i in range(num_features):
        feat_lst = [example[i] for example in data_set]
        unique_vals = set(feat_lst)
        new_ent = 0.0
        for value in unique_vals:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / len(data_set)
            new_ent += prob * calc_shannon_ent(sub_data_set)
        info_gain = base_ent - new_ent
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


def major_cnt(class_list):
    class_cnt = {}

    for vote in class_list:
        if vote not in class_cnt:
            class_cnt[vote] = 0
        class_cnt[vote] += 1
    sorted_class_count = sorted(
        class_cnt.iteritems(), key=operator.itemgetter(1), reverse=True
    )
    return sorted_class_count[0][0]


def create_tree(data_set, labels):
    return create_tree_aux(data_set, labels[:])


def create_tree_aux(data_set, labels):
    class_list = [example[-1] for example in data_set]

    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(data_set[0]) == 1:
        return major_cnt(class_list)

    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]

    decision_tree = {best_feat_label: {}}
    del labels[best_feat]

    feat_values = [example[best_feat] for example in data_set]
    unique_values = set(feat_values)
    for value in unique_values:
        sub_labels = labels[:]
        decision_tree[best_feat_label][value] = create_tree(
            split_data_set(data_set, best_feat, value), sub_labels
        )
    return decision_tree


def classify(tree, labels, vec):
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]

    feat_index = labels.index(first_str)

    for key in second_dict.keys():
        if vec[feat_index] == key:
            if type(second_dict[key]).__name__ == "dict":
                class_label = classify(second_dict[key], labels, vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(in_tree, filename):
    import pickle

    fw = open(filename, "wb")
    pickle.dump(in_tree, fw)
    fw.close()


def get_tree(filename):
    import pickle

    fr = open(filename, "rb")
    tree = pickle.load(fr)
    fr.close()
    return tree