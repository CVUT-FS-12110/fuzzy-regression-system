# Fuzzy regression system

Implementation of **experimental** fuzzy system originaly designed for metal sheet rolling regression task. Trained model is easily interpretable and allows to use not only `predict(...)` but also experimental features `trust(...)` and `confustion(...)`. This can improove decision making by omitting model not trustworthy model outputs.

Please, have in mind that this is merely a PoC.

## Environment

Tested with python `3.10`, install requirements from `requirements.txt`

## Example


```python
from fuzzy.system import FuzzySystem

# create random inputs and targets
x = np.random.rand(1000,4)
theta = np.random.rand(4,1)
y = x @ theta # + np.sum(5*np.sin(x) - 3*np.cos(x) + np.power(x, 2), axis=0)

# train and test split
x_train = x[:800,:]
y_train = y[:800,:]
x_test = x[800:,:]
y_test = y[800:,:]

model = FuzzySystem(
    intervals=[(s, e) for s,e in zip(np.min(x_train, axis=0), np.max(x_train, axis=0))],
    discretization=[5 for _ in range(x_train.shape[1])],
    output_interval = (np.min(y_train, axis=0)[0], np.max(y_train, axis=0)[0]),
    output_discretization = 4,
)

# custom training
c = 0
for i, o in zip(x_train, y_train):
    model.add_rule_io_pair(i, o)
    if c == -1:
        break
    if c%10 == 0:
        logger.info(f"{c}/{x_train.shape[0]} points processed")
    c += 1

```

See file `src/example.py` for simple usage example.