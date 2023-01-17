import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification

# Generate a synthetic dataset for classification
X, y = make_classification(n_features=4, random_state=0)

# Define the MLPClassifier model
clf = MLPClassifier(random_state=0)

# Define a range of learning rates to test
param_grid = {'learning_rate_init': np.logspace(-5, 0, 6)}

# Create a GridSearchCV object to find the optimal learning rate
grid_search = GridSearchCV(clf, param_grid, cv=5)
grid_search.fit(X, y)

# Print the optimal learning rate
print("Optimal learning rate:", grid_search.best_params_['learning_rate_init'])
