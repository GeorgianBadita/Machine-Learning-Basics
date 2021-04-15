import pandas as pd
import numpy as np
import log_reg


def process_data(train_dataset):
    y = np.array([int(x) for x in train_dataset[21]])
    del train_dataset[21]
    x = []
    for _, row in train_dataset.iterrows():
        x.append([1] + [val for val in row])
    x = np.array(x)

    return x, y


DATASET_PATH = 'log_reg/horseColic'

TRAIN_PATH = f"{DATASET_PATH}Training.txt"
TEST_PATH = f"{DATASET_PATH}Test.txt"

train_dataset = pd.read_csv(TRAIN_PATH, header=None, sep='\t')
test_dataset = pd.read_csv(TEST_PATH, header=None, sep='\t')

# TRAIN
x_train, y_train = process_data(train_dataset)
x_train = log_reg.normalize_data(x_train)
##

# TEST
x_test, y_test = process_data(test_dataset)
x_test = log_reg.normalize_data(x_test)

_, final_weights, loss = log_reg.log_reg_stoch(
    x_test, y_test, 0.005, 1_000_00, 10000, True)

print(f"Final weights: {final_weights}")
print(f"Last loss: {loss[-1]}")

print(
    f"Error rate: {log_reg.error_rate(x_test, y_test, final_weights)*100}%")

log_reg.plot_loss(loss)
