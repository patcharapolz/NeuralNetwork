import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from keras.api.models import Sequential
from keras.api.layers import Dense
from keras.api.optimizers import Adam
import pandas as pd

# Create Dataset A and B
A, B = make_blobs(n_samples=200, centers=[[2.0, 2.0], [3.0, 3.0]], cluster_std=0.75, n_features=2, random_state=69)

# Splitting the data into training and testing sets
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.5, random_state=69)

# Creating a neural network model
model = Sequential()
# model.add(Dense(16, input_dim=2, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
optimizer = Adam(0.01)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])

# Training the model
model.fit(A_train, B_train, epochs=300, batch_size=10, verbose=1)

# Makingg predictions
B_pred_prob = model.predict(A_test)
B_pred = np.round(B_pred_prob).astype(int).ravel()

# Plotting the decision boundary
A_min, A_max = A[:, 0].min() - 1, A[:, 0].max() + 1
B_min, B_max = A[:, 1].min() - 1, A[:, 1].max() + 1
AA, BB = np.meshgrid(np.arange(A_min, A_max, 0.1), np.arange(B_min, B_max, 0.1))
Z = model.predict(np.c_[AA.ravel(), BB.ravel()])
Z = np.round(Z).astype(int)
Z = Z.reshape(AA.shape)

plt.contourf(AA, BB, Z, alpha=0.4)
plt.scatter(A[:, 0], A[:, 1], c=B, s=20, edgecolors='k')
plt.xlabel("Feature x1")
plt.ylabel("Feature x2")
plt.title("Decision Boundary")
plt.show()