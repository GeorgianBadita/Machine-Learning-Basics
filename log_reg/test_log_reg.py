import log_reg

DATASET_PATH = "log_reg/test_set.txt"


features, classes = log_reg.read_data_set(DATASET_PATH, add_intercept=True)
features = log_reg.normalize_data(features)


train_x, train_y, test_x, test_y = log_reg.split_data_set(
    features, classes, 80)

_, final_weights, loss = log_reg.log_reg_stoch(
    train_x, train_y, 0.05, 1_000, 100, True)

print(f"Final weights: {final_weights}")
print(f"Last loss: {loss[-1]}")

print(
    f"Error rate: {log_reg.error_rate(test_x, test_y, final_weights)*100}%")

log_reg.plot_2d_decision_boundary(features, classes, final_weights)
log_reg.plot_loss(loss)
