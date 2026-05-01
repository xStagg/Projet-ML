import pickle
import time
from tqdm import tqdm
import numpy as np
import sys

try:
    import cupy as xp
    GPU = True
    print("Using CuPy (GPU)")
except ImportError:
    import numpy as xp
    GPU = False
    print("Using NumPy (CPU)")

# ─────────────────────────────────────────────
# Chargement des données
# ─────────────────────────────────────────────

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def load_cifar10(data_dir="./data/CIFAR-10"):
    batches = [unpickle(f"{data_dir}/data_batch_{i}") for i in range(1, 6)]
    x_train = np.concatenate([b[b'data'] for b in batches], axis=0)
    y_train = np.concatenate([b[b'labels'] for b in batches], axis=0)
    test    = unpickle(f"{data_dir}/test_batch")
    x_test  = test[b'data']
    y_test  = np.array(test[b'labels'])

    # Normalisation + reshape en (N, H, W, C) float32 — CPU d'abord, puis envoi GPU d'un coup
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
    x_test  = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0

    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
    x_train = (x_train - mean) / std
    x_test  = (x_test  - mean) / std

    return (xp.asarray(x_train), xp.asarray(y_train),
            xp.asarray(x_test),  xp.asarray(y_test))

# ─────────────────────────────────────────────
# im2col batché — SANS boucle Python
# X : (B, H, W, C)  →  cols : (B, H_out*W_out, k*k*C)
# ─────────────────────────────────────────────

def im2col_batch(X, k=3, pad=1, stride=1):
    B, H, W, C = X.shape
    # padding sur les axes H, W
    Xp = xp.pad(X, ((0,0),(pad,pad),(pad,pad),(0,0)), mode='constant')
    H_out = (H + 2*pad - k) // stride + 1
    W_out = (W + 2*pad - k) // stride + 1

    # Construire le tenseur col via as_strided (zero-copy)
    shape   = (B, H_out, W_out, k, k, C)
    strides = (Xp.strides[0],
               Xp.strides[1] * stride,
               Xp.strides[2] * stride,
               Xp.strides[1],
               Xp.strides[2],
               Xp.strides[3])
    cols = xp.lib.stride_tricks.as_strided(Xp, shape=shape, strides=strides)
    # (B, H_out*W_out, k*k*C)
    return cols.reshape(B, H_out * W_out, k * k * C)

# ─────────────────────────────────────────────
# Convolution forward batchée
# X      : (B, H, W, C_in)
# filters: (F, k, k, C_in)
# retour : (B, H, W, F)
# ─────────────────────────────────────────────

def conv_forward_batch(X, filters, bias=None):
    B, H, W, C = X.shape
    F, k, _, _ = filters.shape
    cols   = im2col_batch(X, k=k, pad=k//2)           # (B, H*W, k*k*C)
    W_mat  = filters.reshape(F, -1)                    # (F, k*k*C)
    # Einsum: batched matmul (B, H*W, k*k*C) @ (k*k*C, F) → (B, H*W, F)
    Y = cols @ W_mat.T                                 # (B, H*W, F)
    if bias is not None:
        Y += bias                                      # broadcast (F,)
    return Y.reshape(B, H, W, F)                       # (B, H, W, F)

# col2im batché pour le backward conv (dX)
def col2im_batch(dcols, X_shape, k=3, pad=1, stride=1):
    B, H, W, C = X_shape
    H_out = (H + 2*pad - k) // stride + 1
    W_out = (W + 2*pad - k) // stride + 1
    Hp, Wp = H + 2*pad, W + 2*pad
    dXp = xp.zeros((B, Hp, Wp, C), dtype=xp.float32)
    dcols_r = dcols.reshape(B, H_out, W_out, k, k, C)
    for di in range(k):
        for dj in range(k):
            dXp[:, di:di+H_out*stride:stride, dj:dj+W_out*stride:stride, :] += dcols_r[:, :, :, di, dj, :]
    if pad > 0:
        return dXp[:, pad:-pad, pad:-pad, :]
    return dXp

# Backward conv batché
# dY : (B, H, W, F),  X : (B, H, W, C),  filters : (F, k, k, C)
def conv_backward_batch(dY, X, filters):
    B, H, W, F = dY.shape
    k = filters.shape[1]
    cols  = im2col_batch(X, k=k, pad=k//2)            # (B, H*W, k*k*C)
    dY_r  = dY.reshape(B, H*W, F)                     # (B, H*W, F)
    W_mat = filters.reshape(F, -1)                    # (F, k*k*C)

    # dW : somme sur B et positions
    # en fait on veut dW[f] = sum_{b,p} dY_r[b,p,f] * cols[b,p,:]
    dW = xp.einsum('bpf,bpc->fc', dY_r, cols)         # (F, k*k*C)
    dW = dW.reshape(filters.shape)

    db = dY_r.sum(axis=(0, 1))                        # (F,)

    # dcols : (B, H*W, k*k*C)
    dcols = dY_r @ W_mat                               # (B, H*W, k*k*C)
    dX    = col2im_batch(dcols, X.shape, k=k, pad=k//2)
    return dX, dW, db

# ─────────────────────────────────────────────
# MaxPool forward batché — sans boucle Python
# X   : (B, H, W, C)
# retour out : (B, Hp, Wp, C),  mask : (B, H, W, C) bool
# ─────────────────────────────────────────────

def maxpool_forward_batch(X, size=2, stride=2):
    B, H, W, C = X.shape
    Hp = (H - size) // stride + 1
    Wp = (W - size) // stride + 1

    # reshape pour voir chaque fenêtre comme une dimension
    # (B, Hp, stride, Wp, stride, C)  puis max sur les fenêtres
    Xr = X[:, :Hp*stride, :Wp*stride, :]
    Xr = Xr.reshape(B, Hp, stride, Wp, stride, C)
    out = Xr.max(axis=(2, 4))                          # (B, Hp, Wp, C)

    # mask : positions des maxima
    out_exp = out[:, :, xp.newaxis, :, xp.newaxis, :]  # (B, Hp, 1, Wp, 1, C)
    mask_r  = (Xr == out_exp)                          # (B, Hp, stride, Wp, stride, C)
    # gérer les ex-aequo : garder seulement le premier
    mask_r  = mask_r & (xp.cumsum(mask_r.reshape(B, Hp, stride*Wp*stride, C), axis=2).reshape(B, Hp, stride, Wp, stride, C) == 1)
    mask    = mask_r.reshape(B, Hp*stride, Wp*stride, C)
    # reconstruire (B, H, W, C) avec les bords éventuels
    full_mask = xp.zeros((B, H, W, C), dtype=xp.bool_)
    full_mask[:, :Hp*stride, :Wp*stride, :] = mask
    return out, full_mask

# MaxPool backward batché
def maxpool_backward_batch(dout, mask, size=2, stride=2):
    B, H, W, C = mask.shape
    Hp, Wp = dout.shape[1], dout.shape[2]
    dx = xp.zeros((B, H, W, C), dtype=xp.float32)
    # Distribuer le gradient aux positions du max
    dout_exp = dout[:, :, xp.newaxis, :, xp.newaxis, :]   # (B, Hp, 1, Wp, 1, C)
    mask_r   = mask[:, :Hp*stride, :Wp*stride, :].reshape(B, Hp, stride, Wp, stride, C)
    grad_r   = (mask_r * dout_exp).reshape(B, Hp*stride, Wp*stride, C)
    dx[:, :Hp*stride, :Wp*stride, :] = grad_r
    return dx

# ─────────────────────────────────────────────
# Dense + Softmax batché
# X : (B, N_flat),  A : (N_flat, 10),  B_bias : (10,)
# y : (B,) int
# ─────────────────────────────────────────────

def dense_softmax_forward_batch(X, A, B_bias, y):
    Z      = X @ A + B_bias                            # (B, 10)
    Z     -= Z.max(axis=1, keepdims=True)              # stabilité
    expZ   = xp.exp(Z)
    probs  = expZ / expZ.sum(axis=1, keepdims=True)    # (B, 10)
    # Cross-entropy vectorisée
    loss   = -xp.log(probs[xp.arange(len(y)), y] + 1e-15).mean()
    return loss, probs, (X, A, B_bias, probs, y)

def dense_softmax_backward_batch(cache):
    X, A, B_bias, probs, y = cache
    B = len(y)
    dZ            = probs.copy()
    dZ[xp.arange(B), y] -= 1
    dZ           /= B                                  # moyenne sur le batch

    dA       = X.T @ dZ                               # (N_flat, 10)
    dB_bias  = dZ.sum(axis=0)                         # (10,)
    dX       = dZ @ A.T                               # (B, N_flat)
    return dX, dA, dB_bias

# ─────────────────────────────────────────────
# Forward / Backward / Train step — BATCHÉS
# ─────────────────────────────────────────────

def forward_pass_batch(X, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, y):
    # X : (B, 32, 32, 3)
    s1 = conv_forward_batch(X, filters_2d, bias=b1)             # (B, 32, 32, F)
    z1 = xp.maximum(0, s1)

    s2 = conv_forward_batch(z1, filters_3d, bias=b2)            # (B, 32, 32, F)
    z2 = xp.maximum(0, s2)

    p1, mask1 = maxpool_forward_batch(z2)              # (B, 16, 16, F)
    z2p = p1

    s3 = conv_forward_batch(z2p, filters_3d_2, bias=b3)        # (B, 16, 16, F)
    z3 = xp.maximum(0, s3)

    p2, mask2 = maxpool_forward_batch(z3)              # (B, 8, 8, F)

    flat = p2.reshape(len(y), -1)                      # (B, 8*8*F)
    loss, probs, cache_dense = dense_softmax_forward_batch(flat, A, B_bias, y)

    cache = (s1, z1, s2, z2, z2p, mask1, s3, z3, p2, mask2, flat, cache_dense, X)
    return loss, probs, cache

def backward_pass_batch(cache, filters_2d, filters_3d, filters_3d_2):
    s1, z1, s2, z2, z2p, mask1, s3, z3, p2, mask2, flat, cache_dense, X = cache

    # Dense backward
    dflat, dA, dB_bias = dense_softmax_backward_batch(cache_dense)

    # Unflatten
    dp2 = dflat.reshape(p2.shape)

    # Pool2 backward
    dz3 = maxpool_backward_batch(dp2, mask2)

    # ReLU3
    ds3 = dz3 * (s3 > 0)

    # Conv3 backward
    dz2p, dK3, db3 = conv_backward_batch(ds3, z2p, filters_3d_2)

    # Pool1 backward
    dz2 = maxpool_backward_batch(dz2p, mask1)

    # ReLU2
    ds2 = dz2 * (s2 > 0)

    # Conv2 backward
    dz1, dK2, db2 = conv_backward_batch(ds2, z1, filters_3d)

    # ReLU1
    ds1 = dz1 * (s1 > 0)

    # Conv1 backward
    _, dK1, db1 = conv_backward_batch(ds1, X, filters_2d)

    return dA, dB_bias, dK1, db1, dK2, db2, dK3, db3

def train_step_batch(X, y, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, lr=0.001):
    loss, probs, cache = forward_pass_batch(X, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, y)
    dA, dB_bias, dK1, db1, dK2, db2, dK3, db3 = backward_pass_batch(cache, filters_2d, filters_3d, filters_3d_2)

    filters_2d   -= lr * dK1
    b1 -= lr * db1
    filters_3d   -= lr * dK2
    b2 -= lr * db2
    filters_3d_2 -= lr * dK3
    b3 -= lr * db3
    A            -= lr * dA
    B_bias       -= lr * dB_bias
    return loss

def evaluate_model(x_data, y_data,
                  filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3,
                  batch_size=256, max_samples=5000):
    """
    Évalue sur au plus max_samples exemples.
    """
    N = len(y_data)
    if max_samples is not None and max_samples < N:
        idx = xp.random.permutation(N)[:max_samples]
        x_data = x_data[idx]
        y_data = y_data[idx]
        N = max_samples

    n_batches  = N // batch_size
    total_loss = 0.0
    correct    = 0

    for b in range(n_batches):
        Xb = x_data[b*batch_size:(b+1)*batch_size]
        yb = y_data[b*batch_size:(b+1)*batch_size]

        loss, probs, _ = forward_pass_batch(
            Xb, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, yb
        )
        total_loss += float(loss)
        preds       = xp.argmax(probs, axis=1)
        correct    += int((preds == yb).sum())

    rem = N % batch_size
    if rem > 0:
        Xb = x_data[n_batches*batch_size:]
        yb = y_data[n_batches*batch_size:]
        loss, probs, _ = forward_pass_batch(
            Xb, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, yb
        )
        total_loss += float(loss)
        preds       = xp.argmax(probs, axis=1)
        correct    += int((preds == yb).sum())
        n_batches  += 1

    avg_loss = total_loss / n_batches
    accuracy = correct / N * 100.0
    return accuracy, avg_loss

def save_model(path, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3):
    """
    Sauvegarde tous les paramètres dans un fichier .npz.
    Compatible que les poids soient en CuPy ou NumPy.
    """
    # cupy.savez gère automatiquement la conversion vers NumPy en interne
    xp.savez(
        path,
        filters_2d=filters_2d,
        b1=b1,
        filters_3d=filters_3d,
        b2=b2,
        filters_3d_2=filters_3d_2,
        b3=b3,
        A=A,
        B_bias=B_bias,
    )
    print(f"Modèle sauvegardé dans {path}")

def load_model(path):
    """
    Charge les paramètres depuis un .npz et les renvoie
    sous forme de tableaux CuPy (si GPU dispo) ou NumPy.
    """
    # np.load fonctionne, mais cupy.load existe aussi; ici on laisse np.load
    data = np.load(path)
    # Repasser dans le backend actuel xp (NumPy ou CuPy)
    filters_2d   = xp.asarray(data["filters_2d"])
    b1 = xp.asarray(data["b1"])
    filters_3d   = xp.asarray(data["filters_3d"])
    b2 = xp.asarray(data["b2"])
    filters_3d_2 = xp.asarray(data["filters_3d_2"])
    b3 = xp.asarray(data["b3"])
    A            = xp.asarray(data["A"])
    B_bias       = xp.asarray(data["B_bias"])
    return filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage : [nb_epochs] [batch_size] [nb_filters]")
        sys.exit()

    labels_names = ["avion","automobile","oiseau","chat","cerf","chien","grenouille","cheval","bateau","camion"]

    print("Chargement des données...")
    x_train, y_train, x_test, y_test = load_cifar10()
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")

    # Hyperparamètres
    nf         = int(sys.argv[3])
    batch_size = int(sys.argv[2])
    num_epochs = int(sys.argv[1])
    lr         = 0.001
    N          = len(x_train)
    print("Training with : " + "epochs = " + str(num_epochs) + " | " + "batch_size = " + str(batch_size) + " | " + "nb_filters = " + str(nf))

    # Initialisation He
    filters_2d   = xp.random.randn(nf, 3, 3, 3).astype(xp.float32)  * xp.sqrt(2.0 / (3*3*3))
    b1 = xp.zeros(nf, dtype=xp.float32)
    filters_3d   = xp.random.randn(nf, 3, 3, nf).astype(xp.float32) * xp.sqrt(2.0 / (3*3*nf))
    b2 = xp.zeros(nf, dtype=xp.float32)
    filters_3d_2 = xp.random.randn(nf, 3, 3, nf).astype(xp.float32) * xp.sqrt(2.0 / (3*3*nf))
    b3 = xp.zeros(nf, dtype=xp.float32)
    A            = xp.random.randn(nf*8*8, 10).astype(xp.float32)    * xp.sqrt(2.0 / (nf*8*8))
    B_bias       = xp.zeros(10, dtype=xp.float32)

    for epoch in range(num_epochs):
        # Shuffle
        perm = xp.random.permutation(N)
        x_s, y_s = x_train[perm], y_train[perm]

        total_loss = 0.0
        n_batches  = N // batch_size
        t0 = time.perf_counter()

        lr = 0.001 * 0.5 * (1 + xp.cos(xp.pi * epoch / num_epochs))

        for b in tqdm(range(n_batches), desc=f"Epoch {epoch+1}/{num_epochs}"):
            Xb = x_s[b*batch_size:(b+1)*batch_size]
            yb = y_s[b*batch_size:(b+1)*batch_size]
            loss = train_step_batch(Xb, yb, filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3, lr)
            total_loss += float(loss)

        if GPU:
            xp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()

        avg_loss = total_loss / n_batches
        imgs_per_sec = N / (t1 - t0)
        print(f"  loss={avg_loss:.4f}  |  {imgs_per_sec:.0f} img/s  |  {t1-t0:.1f}s")
    
        save_model("params_cifar_conv_gpu.npz",
            filters_2d, filters_3d, filters_3d_2, A, B_bias, b1, b2, b3)
