import logging

import numpy as np

from fuzzy.system import FuzzySystem

logger = logging.getLogger("fuzzymodel")
logging.basicConfig(level=logging.INFO)


# create random inputs and targets
x = np.random.rand(1000,4)
theta = np.random.rand(4,1)
y = x @ theta # + np.sum(5*np.sin(x) - 3*np.cos(x) + np.power(x, 2), axis=0)

print(y)

# train and test split
x_train = x[:800,:]
y_train = y[:800,:]
x_test = x[800:,:]
y_test = y[800:,:]

logger.info(f"Data, loaded:\n   train: {x_train.shape}, {y_train.shape}\n    test: {x_test.shape} {y_test.shape}")

model = FuzzySystem(
    intervals=[(s, e) for s,e in zip(np.min(x_train, axis=0), np.max(x_train, axis=0))],
    discretization=[5 for _ in range(x_train.shape[1])],
    output_interval = (np.min(y_train, axis=0)[0], np.max(y_train, axis=0)[0]),
    output_discretization = 4,
)

c = 0
for i, o in zip(x_train, y_train):
    model.add_rule_io_pair(i, o)
    if c == -1:
        break
    if c%10 == 0:
        logger.info(f"{c}/{x_train.shape[0]} points processed")
    c += 1

logger.info(f"Number of rules: {len(model.rules)}")

error = []
for xrow, y in zip(x_test, y_test):
    r = model.predict(xrow)
    error.append(r-y[0])
    
# example of last test row
logger.info(f"Model output: {r}, correct output {y[0]}, error: {r-y[0]} ")
logger.info(f"Trust: {model.trust(xrow)} confusion {model.confusion(xrow)}")
    
# print statistics
error = np.array(error)

print(f"Metrics | MAE: {np.mean(np.abs(error))}, std: {np.std(error)}")
    

