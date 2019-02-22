from utils import *

dataset = "cora"
adj, labels, features, y_train, y_val, y_test, initial_train_mask, val_mask, test_mask = load_data(dataset)
  