import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_spiral(n_points=200, noise=0.1, seed=42):
    np.random.seed(seed)
    n = n_points // 2
    theta0 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise
    theta1 = np.linspace(0, 4 * np.pi, n) + np.random.randn(n) * noise + np.pi
    r = np.linspace(0.1, 1.0, n)
    X0 = np.c_[r * np.cos(theta0), r * np.sin(theta0)]
    X1 = np.c_[r * np.cos(theta1), r * np.sin(theta1)]
    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(n), np.ones(n)])
    return X, y

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    return (x > 0).astype(float)

def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


X, y = generate_spiral(n_points=400, noise=0.15, seed=42)
input_dim = 2
hidden1 = 64
hidden2 = 64
output_dim = 1
np.random.seed(42)
W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
b1 = np.random.randn(hidden1) * 0.01
W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
b2 = np.random.randn(hidden2) * 0.01
W3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
b3 = np.random.randn(output_dim) * 0.01

lr = 0.1
n_epochs = 10000
losses = []

for epoch in range(n_epochs):
    z1 = X @ W1 + b1  
    a1 = relu(z1)
    z2 = a1 @ W2 + b2  
    a2 = relu(z2)
    z3 = a2 @ W3 + b3  
    y_pred = sigmoid(z3).flatten()

   
    loss = bce_loss(y, y_pred)
    losses.append(loss)

    err3 = y_pred - y 
    dW3 = a2.T @ err3.reshape(-1, 1) / len(y)
    db3 = np.mean(err3)
    err2 = (err3.reshape(-1, 1) @ W3.T) * relu_grad(z2)
    dW2 = a1.T @ err2 / len(y)
    db2 = np.mean(err2, axis=0)
    err1 = (err2 @ W2.T) * relu_grad(z1)
    dW1 = X.T @ err1 / len(y)
    db1 = np.mean(err1, axis=0)

    W3 -= lr * dW3
    b3 -= lr * db3
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 500 == 0:
        acc = np.mean((y_pred > 0.5) == y)
        print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc:.2%}")


h = 0.02
xx, yy = np.meshgrid(
    np.arange(X[:, 0].min() - 0.2, X[:, 0].max() + 0.2, h),
    np.arange(X[:, 1].min() - 0.2, X[:, 1].max() + 0.2, h)
)
grid = np.c_[xx.ravel(), yy.ravel()]
z1g = np.dot(grid, W1) + b1
a1g = relu(z1g)
z2g = np.dot(a1g, W2) + b2
a2g = relu(z2g)
zg = sigmoid(np.dot(a2g, W3) + b3).reshape(xx.shape)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].contourf(xx, yy, zg, alpha=0.4, cmap='RdBu')
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', s=30, edgecolors='k')
axes[0].set_title("Frontière de décision (2‑64‑64‑1)")
axes[1].plot(losses)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss BCE")
axes[1].set_title("Courbe de loss spirale")
plt.savefig("phase4_spirale.png", dpi=100, bbox_inches='tight')

print(f"\nLoss finale : {losses[-1]:.4f}")
print(f"Accuracy finale : {np.mean((y_pred > 0.5) == y):.2%}")
