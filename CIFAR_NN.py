import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import pickle
    from PIL import Image

    return F, mo, nn, np, torch, tqdm


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 style="text-align:center;"><b>Neural Network</b></h1>
    """)
    return


@app.cell
def _(F, nn):
    class MLP(nn.Module):
        def __init__(self, H=3):
            super().__init__()
            self.input_layer = nn.Linear(1024, 512)
            layers = []
            layers.append(nn.Linear(512, 256))
            layers.append(nn.Linear(256, 128))
            self.hidden_layers = nn.ModuleList(layers)
            self.output_layer = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.input_layer(x))
            for layer in self.hidden_layers:
                x = F.relu(layer(x))
            return self.output_layer(x)

    return (MLP,)


@app.cell
def _(MLP, nn, torch, x_test_gray, x_train_gray, y_test, y_train):
    model = MLP()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    x_train_tensor = torch.tensor(x_train_gray, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test_gray, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    return (
        criterion,
        model,
        optimizer,
        x_test_tensor,
        x_train_tensor,
        y_test_tensor,
        y_train_tensor,
    )


@app.cell
def _(torch, tqdm):
    def train(model, criterion, optimizer, x_train, y_train, x_test, y_test, epochs=10):
        for epoch in range(epochs):
            for i in tqdm(range(0, len(x_train), 64)):
                x_batch = x_train[i : i + 64]
                y_batch = y_train[i : i + 64]
                model.train()
                optimizer.zero_grad()
                output = model(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    test_output = model(x_test)
                    test_loss = criterion(test_output, y_test)
                    pred = test_output.argmax(dim=1)
                    accuracy = (pred == y_test).float().mean().item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy:.4f}")

    return (train,)


@app.cell
def _(
    criterion,
    model,
    optimizer,
    train,
    x_test_tensor,
    x_train_tensor,
    y_test_tensor,
    y_train_tensor,
):
    train(model, criterion, optimizer, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor)
    return


if __name__ == "__main__":
    app.run()
