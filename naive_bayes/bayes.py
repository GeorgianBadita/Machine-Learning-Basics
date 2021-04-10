from naive_bayes import data
import random
import numpy as np


def train_nb(train_matrix, train_category):
    num_tarin_lines = len(train_matrix)
    num_words = len(train_matrix[0])

    p_class1 = sum(train_category) / num_tarin_lines
    p0_num = np.ones(num_words)
    p0_denom = 2
    p1_num = np.ones(num_words)
    p1_denom = 2

    for i in range(num_tarin_lines):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vect = np.log(p1_num / p1_denom)
    p0_vect = np.log(p0_num / p0_denom)

    return p0_vect, p1_vect, p_class1


def classify_nb(to_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(to_classify * p1_vec) + np.log(p_class1)
    p0 = sum(to_classify * p0_vec) + np.log(1 - p_class1)

    return 1 if p1 > p0 else 0


def classify_posts():
    posts, classes = data.load_data_set_blog()
    vocab = data.create_vocab_list(posts)

    train_matrix = []
    for post in posts:
        train_matrix.append(data.set_of_words_2_vec(vocab, post))

    p0_v, p1_v, p_ab = train_nb(train_matrix, classes)

    test_entry_non_abusive = ["love", "my", "dalmatian"]

    post_non_a = np.array(data.set_of_words_2_vec(
        vocab, test_entry_non_abusive))
    print(
        f"Entry: {test_entry_non_abusive} classified as: {classify_nb(post_non_a, p0_v, p1_v, p_ab)}"
    )

    test_entry_abusive = ["stupid", "garbage"]
    post_a = np.array(data.set_of_words_2_vec(vocab, test_entry_abusive))
    print(
        f"Entry: {test_entry_abusive} classified as: {classify_nb(post_a, p0_v, p1_v, p_ab)}"
    )


def read_email(path, doc_list, full_text, class_list, class_value):
    word_list = data.text_parse(open(path, "r", encoding='latin-1').read())
    doc_list.append(word_list)
    full_text.extend(word_list)
    class_list.append(class_value)


def spam_filtering():
    doc_list = []
    class_list = []
    full_text = []

    for i in range(1, 26):
        read_email(f'naive_bayes/email/spam/{i}.txt',
                   doc_list, full_text, class_list, 1)
        read_email(f'naive_bayes/email/ham/{i}.txt',
                   doc_list, full_text, class_list, 0)

    vocab = data.create_vocab_list(doc_list)
    training_set = list(range(50))
    test_set = []

    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del training_set[rand_index]

    train_mat = []
    train_classes = []
    for index in training_set:
        train_mat.append(data.set_of_words_2_vec(vocab, doc_list[index]))
        train_classes.append(class_list[index])

    p0_v, p1_v, p_spam = train_nb(np.array(train_mat), np.array(train_classes))

    error_cnt = 0
    for index in test_set:
        word_vec = data.set_of_words_2_vec(vocab, doc_list[index])
        if classify_nb(np.array(word_vec), p0_v, p1_v, p_spam) != class_list[index]:
            error_cnt += 1
    print(f'Erorr rate is: {error_cnt/len(test_set)}')
    return error_cnt/len(test_set)


def test_spam_more_runs(runs=200):
    err_rate = sum([spam_filtering() for _ in range(runs)])/runs

    print(f'Ovearl error rate: {err_rate}')


test_spam_more_runs()
