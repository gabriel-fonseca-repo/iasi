import numpy as np
import matplotlib.pyplot as plt

X = np.random.uniform(low=5, high=30, size=(10, 3)).astype(int)
y = np.random.uniform(size=(10, 1))
B = np.random.uniform(size=(3, 1))

t1 = y.T @ X @ B
t2 = (X @ B).T @ y
t3 = B.T @ X.T @ y

M = X.T @ X

print(t1, end="\n\n")
print(t2, end="\n\n")
print(t3, end="\n\n")
print(M, end="\n\n")

plt.scatter(x, y, color="darkred")
plt.grid(True)
plt.xlim(0, 1150)
x.shape(10, 1)
y.shape(10, 1)
X = np.concatenate((np.ones(10, 1), x), axis=1)
b_hat = np.linalg.inv(X.T @ X) @ X.T @ y
b_hat1 = np.linalg.pinv(X.T @ X) @ X.T @ y
b_hat2 = np.linalg.lstsq(X, y)[0]
x_axis = np.linspace(-10, 1200, 1200).reshape(1200, 1)
X_AXIS = np.concatenate((np.ones(1200, 1), x_axis), axis=1)
y_hat = X_AXIS @ b_hat
plt.plot(x_axis, y_hat, color="green", linewidth=2)
x_new = 400
y_hat1 = np.array([[1, x_new]]) @ b_hat
plt.scatter(x_new, y_hat1[0, 0], color="blue", marker="1", s=90)

plt.show()
