import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

DATASET_NAME = "breast_cancer"

data = load_breast_cancer()
X, y = data.data.astype(np.float64), data.target.astype(np.float64)
print(f"Dataset : {DATASET_NAME}")
print(f"Shape X : {X.shape} | Classes : {np.unique(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_features = X_train.shape[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(float)


def bce_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def accuracy(y_true, y_pred):
    return np.mean((y_pred > 0.5) == y_true)


np.random.seed(42)
W1 = np.random.randn(n_features, 16) * np.sqrt(2.0 / n_features)
b1 = np.zeros(16)
W2 = np.random.randn(16, 8) * np.sqrt(2.0 / 16)
b2 = np.zeros(8)
W3 = np.random.randn(8, 1) * np.sqrt(2.0 / 8)
b3 = np.zeros(1)

lr_numpy = 0.1
n_epochs_numpy = 200
numpy_losses = []

for epoch in range(n_epochs_numpy):
    z1 = X_train @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = relu(z2)
    z3 = a2 @ W3 + b3
    y_pred_train = sigmoid(z3).flatten()

    loss = bce_loss(y_train, y_pred_train)
    numpy_losses.append(loss)

    err3 = y_pred_train - y_train
    dW3 = a2.T @ err3.reshape(-1, 1) / len(y_train)
    db3 = np.mean(err3)
    err2 = (err3.reshape(-1, 1) @ W3.T) * relu_grad(z2)
    dW2 = a1.T @ err2 / len(y_train)
    db2 = np.mean(err2, axis=0)
    err1 = (err2 @ W2.T) * relu_grad(z1)
    dW1 = X_train.T @ err1 / len(y_train)
    db1 = np.mean(err1, axis=0)

    W3 -= lr_numpy * dW3
    b3 -= lr_numpy * db3
    W2 -= lr_numpy * dW2
    b2 -= lr_numpy * db2
    W1 -= lr_numpy * dW1
    b1 -= lr_numpy * db1

z1_t = X_test @ W1 + b1
a1_t = relu(z1_t)
z2_t = a1_t @ W2 + b2
a2_t = relu(z2_t)
y_pred_test_np = sigmoid(a2_t @ W3 + b3).flatten()
numpy_test_loss = bce_loss(y_test, y_pred_test_np)
numpy_test_acc = accuracy(y_test, y_pred_test_np)

print(f"\nNumpy from-scratch | Loss finale : {numpy_losses[-1]:.4f} | Test accuracy : {numpy_test_acc:.4f}")

tf.random.set_seed(42)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(n_features,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'],
)

start_keras = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=0,
)
keras_time = time.time() - start_keras

keras_test_loss, keras_test_acc = model.evaluate(X_test, y_test, verbose=0)
keras_val_losses = history.history['val_loss']

print(f"Keras | Loss finale : {keras_test_loss:.4f} | Test accuracy : {keras_test_acc:.4f}")
print(f"Temps d'entrainement Keras : {keras_time:.1f}s")

gain = (keras_test_acc - numpy_test_acc) * 100
print(f"Gain Keras vs Numpy : {gain:+.1f} points de %")

print("\n=== TABLEAU COMPARATIF ===")
print(f"{'Pipeline':20s} | {'Loss test':12s} | {'Test accuracy':14s}")
print("-" * 52)
print(f"{'Numpy from-scratch':20s} | {numpy_test_loss:.4f} | {numpy_test_acc:.4f}")
print(f"{'Keras':20s} | {keras_test_loss:.4f} | {keras_test_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

epochs_np = range(1, n_epochs_numpy + 1)
epochs_k = range(1, len(keras_val_losses) + 1)
axes[0].plot(epochs_np, numpy_losses, label='Numpy (train loss)', linewidth=2)
axes[0].plot(epochs_k, keras_val_losses, label='Keras (val loss)', linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].set_title("Courbes de loss")
axes[0].legend()

pipelines = ['Numpy', 'Keras']
accs = [numpy_test_acc, keras_test_acc]
axes[1].bar(pipelines, accs, color=['#4C72B0', '#DD8452'])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel("Test accuracy")
axes[1].set_title("Accuracy sur le jeu de test")
for i, v in enumerate(accs):
    axes[1].text(i, v + 0.02, f"{v:.3f}", ha='center')

plt.tight_layout()
plt.savefig("phase8_comparaison.png", dpi=100, bbox_inches='tight')
print("\nGraphique sauvegarde : phase8_comparaison.png")

X_extreme = np.full((1, n_features), 99999.0)
X_extreme_scaled = scaler.transform(X_extreme)
pred_extreme_np = sigmoid(
    relu(relu(X_extreme_scaled @ W1 + b1) @ W2 + b2) @ W3 + b3
).item()
pred_extreme_keras = model.predict(X_extreme_scaled, verbose=0)[0, 0]
print(f"\n[Hors distribution] Prediction numpy : {pred_extreme_np:.4f}")
print(f"[Hors distribution] Prediction Keras  : {pred_extreme_keras:.4f}")
print("Une confiance elevee sur des donnees aberrantes est un signal d'alarme en production.")