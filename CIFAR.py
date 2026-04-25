import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pickle
    from PIL import Image

    return np, plt, tqdm


@app.cell
def _(np):
    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    data_batch_1 = unpickle("./data/CIFAR-10/data_batch_1")
    data_batch_2 = unpickle("./data/CIFAR-10/data_batch_2")
    data_batch_3 = unpickle("./data/CIFAR-10/data_batch_3")
    data_batch_4 = unpickle("./data/CIFAR-10/data_batch_4")
    data_batch_5 = unpickle("./data/CIFAR-10/data_batch_5")
    test_batch = unpickle("./data/CIFAR-10/test_batch")
    x_train = np.concatenate((data_batch_1[b'data'], data_batch_2[b'data'], data_batch_3[b'data'], data_batch_4[b'data'], data_batch_5[b'data']), axis=0)
    y_train = np.concatenate((data_batch_1[b'labels'], data_batch_2[b'labels'], data_batch_3[b'labels'], data_batch_4[b'labels'], data_batch_5[b'labels']), axis=0)
    x_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])
    return x_test, x_train, y_test, y_train


@app.cell
def _(x_test, x_train, y_test, y_train):
    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)
    return


@app.cell
def _(np):
    def rgb_to_bw(img):
        weights = np.array([0.299, 0.587, 0.114])
        gray = np.dot(img[..., :3], weights)/255.0
        return gray.astype(np.float32)

    return (rgb_to_bw,)


@app.cell
def _(np, rgb_to_bw, x_test, x_train):
    print("x_train shape before conversion:", x_train.shape)
    print("x_test shape before conversion:", x_test.shape)

    x_train_gray = np.array([rgb_to_bw(x_train[i].reshape(3, 32, 32).transpose(1, 2, 0)) for i in range(x_train.shape[0])])
    mean = x_train_gray.mean()  # scalaire unique (plus de canaux RGB)
    std = x_train_gray.std()
    x_train_gray = ((x_train_gray - mean) / std).reshape(x_train_gray.shape[0], -1)

    x_test_gray = np.array([rgb_to_bw(x_test[i].reshape(3, 32, 32).transpose(1, 2, 0)) for i in range(x_test.shape[0])])
    mean = x_test_gray.mean()  # scalaire unique (plus de canaux RGB)
    std = x_test_gray.std()
    x_test_gray = ((x_test_gray - mean) / std).reshape(x_test_gray.shape[0], -1)

    print("x_train shape after conversion:", x_train_gray.shape)
    print("x_test shape after conversion:", x_test_gray.shape)
    return x_test_gray, x_train_gray


@app.cell
def _(plt, x_test, x_test_gray, y_train):
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    axes[0].imshow(x_test[0].reshape(3, 32, 32).transpose(1, 2, 0))
    axes[0].set_title(f"Label: {y_train[0]}")
    axes[0].axis('off')
    axes[1].imshow(x_test_gray[1].reshape(32, 32), cmap='gray')
    axes[1].set_title(f"Label: {y_train[0]}")
    axes[1].axis('off')
    plt.show()
    return


@app.cell
def _(plt, predict):
    def show_predicted_img(A, B, img, true_label):
        img = img.reshape(1, -1)
        label = true_label
        pred_label = predict(A, B, img)[0]
        plt.imshow(img.reshape(32, 32), cmap='gray')
        plt.title(f"True: {label}, Pred: {pred_label}")
        plt.axis('off')
        plt.show()

    return (show_predicted_img,)


@app.cell
def _(x_train_gray):
    print("x_train_gray:", x_train_gray[0])
    return


@app.cell
def _(x_train_gray):
    print(x_train_gray.max())
    print(x_train_gray.mean())
    print(x_train_gray.std())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 style="text-align:center;"><b>Modèle Linéaire</b></h1>
    """)
    return


@app.cell
def _(np):
    def forward(A, B, X):
        logits = A @ X + B  # X: (D, B), A: (10, D), B: (10, 1)
        logits = logits - logits.max(axis=0, keepdims=True)  # (10, B)
        exp = np.exp(logits)  # stabilité numérique
        return exp / exp.sum(axis=0, keepdims=True)  # softmax (10, B)

    return (forward,)


@app.cell
def _(np):
    def cross_entropy_loss(probs, y):
        N = y.shape[0]
        # Clip pour éviter log(0)
        log_p = -np.log(probs[np.arange(N), y] + 1e-9)
        return log_p.mean()

    return (cross_entropy_loss,)


@app.cell
def _(np):
    def softmax(z):
        # Stabilité numérique : soustraction du max
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    return (softmax,)


@app.cell
def _(np):
    def predict(A, B, X):
        # X : (N, 1024), A : (1024, 10), B : (10,)
        # print("X :", X.shape, " / A :", A.shape, " / B :", B.shape)
        logits = X @ A + B       # (N, 10)
        return np.argmax(logits, axis=1)

    return (predict,)


@app.cell
def _(cross_entropy_loss, np, predict, softmax):
    def train_2(X_train, y_train, lr=0.1, epochs=30, batch_size=256):
        print(X_train.shape)
        N, D = X_train.shape
        K = 10
        # Initialisation petite mais non nulle
        A = np.random.randn(D, K) * 0.01
        B = np.zeros(K)

        for epoch in range(epochs):
            # Shuffle
            idx = np.random.permutation(N)
            X_train, y_train = X_train[idx], y_train[idx]

            total_loss = 0
            for i in range(0, N, batch_size):
                X_b = X_train[i:i+batch_size]   # (bs, 1024)
                y_b = y_train[i:i+batch_size]   # (bs,)
                bs = X_b.shape[0]

                # Forward
                logits = X_b @ A + B            # (bs, 10)
                probs  = softmax(logits)         # (bs, 10)
                loss   = cross_entropy_loss(probs, y_b)
                total_loss += loss

                # ⬇️ Gradient : c'est ICI que les gens font des erreurs
                dL_dz = probs.copy()
                dL_dz[np.arange(bs), y_b] -= 1  # (bs, 10)
                dL_dz /= bs                      # moyenne sur le batch

                # Gradients des paramètres
                dA = X_b.T @ dL_dz              # (1024, 10)
                dB = dL_dz.sum(axis=0)          # (10,)

                # Mise à jour
                A -= lr * dA
                B -= lr * dB

            preds = predict(A, B, X_train)
            acc = (preds == y_train).mean() * 100
            n_batches = N // batch_size
            print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/n_batches:.4f} | Train Acc: {acc:.2f}%")

        return A, B

    return (train_2,)


@app.cell
def _(forward, log_loss, np, tqdm):
    def train(A, B, x_train, y_train, epochs=10, lr=0.01):
        n = x_train.shape[0]
        X_all = x_train.reshape(n, -1).T
        Y_all = np.eye(10)[y_train].T
        total_iters = epochs * n
        with tqdm(total=total_iters, desc='Training') as pbar:
            for epoch in range(epochs):  # Précalcul hors boucle — shape (n, D) et (10, n)
                Y_pred = forward(A, B, X_all)  # (D, n)
                loss = log_loss(Y_all, Y_pred)  # (10, n)
                dO = Y_pred - Y_all
                dA = dO @ X_all.T / n
                dB = dO.mean(axis=1, keepdims=True)
                A = A - lr * dA
                B = B - lr * dB  # Forward vectorisé sur tout le dataset
                pbar.update(n)  # (10, n)
                pbar.set_postfix({'epoch': f'{epoch + 1}/{epochs}', 'loss': f'{loss:.4f}'})  # Loss  # Backpropagation vectorisée  # (10, n)  # (10, D)  # (10, 1)

    return


@app.cell
def _(mo):
    run = mo.ui.run_button(label="Start Training")
    arr = mo.ui.array([
        mo.ui.slider(0.01, 0.1, 0.001, label="Learning Rate"),
        mo.ui.slider(10, 1000, 10, label="Epochs")
    ])
    arr, run
    return arr, run


@app.cell
def _(arr, run, train_2, x_train_gray, y_train):
    A, B = None, None
    if run.value:
        A, B = train_2(x_train_gray, y_train, epochs=arr.value[1], lr=arr.value[0])
        print('Training completed.')
    return A, B


@app.cell
def _(A, B, np, predict, x_test_gray, y_test):
    predictions = predict(A, B, x_test_gray)
    accuracy = np.mean(predictions == y_test)
    print(f"Test Accuracy: {accuracy:.4f}")
    return


@app.cell
def _(A, B, show_predicted_img, x_test_gray, y_test):
    show_predicted_img(A, B, x_test_gray[1], y_test[1])
    return


if __name__ == "__main__":
    app.run()
