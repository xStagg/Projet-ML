import pickle
from tqdm import tqdm

try:
    import cupy as xp
    print("Using CuPy (GPU)")
except ImportError:
    import numpy as xp
    print("Using NumPy (CPU)")


def to_xp(a):
    return xp.asarray(a)
    

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batch_1 = unpickle("./data/CIFAR-10/data_batch_1")
data_batch_2 = unpickle("./data/CIFAR-10/data_batch_2")
data_batch_3 = unpickle("./data/CIFAR-10/data_batch_3")
data_batch_4 = unpickle("./data/CIFAR-10/data_batch_4")
data_batch_5 = unpickle("./data/CIFAR-10/data_batch_5")
test_batch = unpickle("./data/CIFAR-10/test_batch")
x_train = xp.concatenate((
    to_xp(data_batch_1[b'data']),
    to_xp(data_batch_2[b'data']),
    to_xp(data_batch_3[b'data']),
    to_xp(data_batch_4[b'data']),
    to_xp(data_batch_5[b'data'])), axis=0)
y_train = xp.concatenate((
    to_xp(data_batch_1[b'labels']),
    to_xp(data_batch_2[b'labels']),
    to_xp(data_batch_3[b'labels']),
    to_xp(data_batch_4[b'labels']),
    to_xp(data_batch_5[b'labels'])), axis=0)
x_test = test_batch[b'data']
y_test = xp.array(test_batch[b'labels'])
labels = ["avion", "automobile", "oiseau", "chat", "cerf", "chien", "grenouille", "cheval", "bateau", "camion"]

def conv3d_backward_simple(dY, image, filters, padding=1):
    # dY : (F, H, W) = dL/ds
    H, W, C = image.shape
    F, kh, kw, Ck = filters.shape
    assert C == Ck

    # padding de l'entrée
    x_padded = pad_image(image, padding)
    dx_padded = xp.zeros_like(x_padded, dtype=xp.float32)
    dK = xp.zeros_like(filters, dtype=xp.float32)
    db = dY.sum(axis=(1, 2))   # (F,)

    for i in range(H):
        for j in range(W):
            patch = x_padded[i:i+kh, j:j+kw, :]  # (kh, kw, C)
            for f in range(F):
                grad_val = dY[f, i, j]
                # dL/dK_f += grad_val * patch
                dK[f] += grad_val * patch
                # dL/dx_padded += grad_val * K_f
                dx_padded[i:i+kh, j:j+kw, :] += grad_val * filters[f]

    dx = dx_padded[padding:-padding, padding:-padding, :]
    return dx, dK, db

def im2col(image, k=3):
    H, W, C = image.shape
    pad = k // 2
    padded = pad_image(image, pad)
    cols = xp.zeros((H * W, k * k * C), dtype=xp.float32)
    idx = 0
    for i in range(H):
        for j in range(W):
            patch = padded[i:i+k, j:j+k, :].reshape(-1)
            cols[idx] = patch
            idx += 1
    return cols  # (H*W, 9*C)

def conv3d_batch(image, filters, bias=None):
    H, W, C = image.shape
    F, kh, kw, Ck = filters.shape
    assert C == Ck

    X_col = im2col(image, k=kh)           # (H*W, 9*C)
    W_mat = filters.reshape(F, -1)            # (F, 9*C)

    Y = W_mat @ X_col.T                       # (F, H*W)
    if bias is not None:
        Y += bias.reshape(F, 1)
    Y = Y.reshape(F, H, W)               # (F, H, W)
    return Y

def dense_softmax_forward(flatten_res, A, B, y_true):
    Z = flatten_res @ A + B          # (10,)
    Z_shift = Z - xp.max(Z)
    exp_Z = xp.exp(Z_shift)
    probs = exp_Z / xp.sum(exp_Z)    # (10,)
    loss = -xp.log(probs[y_true] + 1e-15)  # Cross-entropy loss
    cache = (flatten_res, A, B, Z, probs, y_true)
    return loss, probs, cache

def flatten_backward(d_flatten, shape):
    return d_flatten.reshape(shape)

def dense_softmax_backward(cache):
    flatten_res, A, B, Z, probs, y_true = cache
    dZ = probs.copy()
    dZ[y_true] -= 1  # (10,)

    # gradients paramètres
    dA = xp.outer(flatten_res, dZ)  # (N, 10)
    dB = dZ  # (10,)

    # gradients entrée
    d_flatten = A @ dZ  # (N,)

    return d_flatten, dA, dB

def pad_image(image, padding):
    H, W, C = image.shape
    x_padded = xp.zeros((H + 2*padding, W + 2*padding, C), dtype=image.dtype)
    x_padded[padding:padding+H, padding:padding+W, :] = image
    return x_padded

def max_pool2d_forward(x, size=2, stride=2):
    H, W = x.shape
    Hp = (H - size) // stride + 1
    Wp = (W - size) // stride + 1

    out = xp.zeros((Hp, Wp), dtype=xp.float32)
    mask = xp.zeros((H, W), dtype=xp.bool_)

    for i in range(Hp):
        for j in range(Wp):
            i0 = i * stride
            j0 = j * stride

            max_val = x[i0, j0]
            max_u = 0
            max_v = 0

            for u in range(size):
                for v in range(size):
                    val = x[i0 + u, j0 + v]
                    if val > max_val:
                        max_val = val
                        max_u = u
                        max_v = v

            out[i, j] = max_val
            mask[i0 + max_u, j0 + max_v] = True

    return out, mask

def max_pool2d_backward(dout, mask, size=2, stride=2):
    H, W = mask.shape
    Hp, Wp = dout.shape
    dx = xp.zeros_like(mask, dtype=xp.float32)

    for i in range(Hp):
        for j in range(Wp):
            # gradient ne va qu'à la position du max
            block_mask = mask[i*stride:i*stride+size, j*stride:j*stride+size]
            dx[i*stride:i*stride+size, j*stride:j*stride+size] += block_mask * dout[i, j]
    return dx

def forward_pass(image, filters_2d, filters_3d, filters_3d_2, A, B, label):
    # ---- FORWARD ----
    # 1) conv1
    s1 = conv3d_batch(image, filters_2d)          # (F1, 32, 32)
    s1 = xp.transpose(s1, (1, 2, 0))               # (32, 32, F1)

    # 2) ReLU1
    z1 = xp.maximum(0, s1)

    # 3) conv2
    s2 = conv3d_batch(z1, filters_3d)             # (F2, 32, 32)
    s2 = xp.transpose(s2, (1, 2, 0))              # (32, 32, F2)

    # 4) ReLU2
    z2 = xp.maximum(0, s2)

    # 5) MaxPool1 (32 -> 16)
    maxpooled_3d_images_batch = xp.zeros((z2.shape[0]//2,
                                        z2.shape[1]//2,
                                        z2.shape[2]), dtype=xp.float32)
    masks_pool1 = []
    for c in range(z2.shape[2]):
        pooled_c, mask_c = max_pool2d_forward(z2[:, :, c], size=2, stride=2)
        maxpooled_3d_images_batch[:, :, c] = pooled_c
        masks_pool1.append(mask_c)            # (32,32) bool

    # 6) conv3
    s3 = conv3d_batch(maxpooled_3d_images_batch, filters_3d_2)  # (F3, 16, 16)
    s3 = xp.transpose(s3, (1, 2, 0))              # (16, 16, F3)

    # 7) ReLU3
    z3 = xp.maximum(0, s3)

    # 8) MaxPool2 (16 -> 8)
    maxpooled_3d_images_batch_2 = xp.zeros((z3.shape[0]//2,
                                            z3.shape[1]//2,
                                            z3.shape[2]), dtype=xp.float32)
    masks_pool2 = []
    for c in range(z3.shape[2]):
        pooled_c2, mask_c2 = max_pool2d_forward(z3[:, :, c], size=2, stride=2)
        maxpooled_3d_images_batch_2[:, :, c] = pooled_c2
        masks_pool2.append(mask_c2)           # (16,16) bool

    # 9) Dense + Softmax
    flatten_result = maxpooled_3d_images_batch_2.flatten()  # (N,)

    loss, probs, cache_dense = dense_softmax_forward(flatten_result, A, B, label)
    return loss, probs, cache_dense, masks_pool1, masks_pool2, z1, z2, z3, s1, s2, s3, maxpooled_3d_images_batch, maxpooled_3d_images_batch_2

def backward_pass(image, cache_dense, masks_pool1, masks_pool2, z1, z2, z3, s1, s2, s3, maxpooled_3d_images_batch, maxpooled_3d_images_batch_2):
    # ---- BACKWARD ----

    # 1) Backward dense + softmax ->
    d_flatten, dA, dB = dense_softmax_backward(cache_dense)

    # 2) Unflatten vers sortie 2e max pool -> (8, 8, 64)
    d_last_pool = flatten_backward(d_flatten, maxpooled_3d_images_batch_2.shape)

    # 3) Backward du 2e max pool -> d_L/d_z3
    d_z3 = xp.zeros_like(z3)
    for _i in range(z3.shape[2]):
        d_z3[:, :, _i] = max_pool2d_backward(d_last_pool[:, :, _i], masks_pool2[_i], size=2, stride=2)

    # 4) Backward ReLU3 -> d_L/d_s3
    d_s3 = d_z3 * (s3 > 0)  # (16, 16, 64)

    # 5) Backward conv3 -> d_z2_pool, dK3, db3
    d_s3_conv = d_s3.transpose(2, 0, 1)  # (64, 16, 16)
    d_z2_pool, dK3, db3 = conv3d_backward_simple(d_s3_conv, maxpooled_3d_images_batch, filters_3d_2)

    # 6) Backward 1er max pool -> d_L/d_z2
    d_z2 = xp.zeros_like(z2)
    for _i in range(z2.shape[2]):
        d_z2[:, :, _i] = max_pool2d_backward(d_z2_pool[:, :, _i], masks_pool1[_i], size=2, stride=2)

    # 7) Backward ReLU2 -> d_L/d_s2
    d_s2 = d_z2 * (s2 > 0)  # (32, 32, 64)

    # 8) Backward conv2 -> d_z1, dK2, db2
    d_s2_conv = d_s2.transpose(2, 0, 1)  # (64, 32, 32)
    d_z1_pool, dK2, db2 = conv3d_backward_simple(d_s2_conv, z1, filters_3d)

    # 9) Backward ReLU1 -> d_L/d_s1
    d_s1 = d_z1_pool * (s1 > 0)  # (32, 32, 64)

    # 10) Backward conv1 -> d_x, dK1, db1
    d_s1_conv = d_s1.transpose(2, 0, 1)  # (64, 32, 32)
    d_x, dK1, db1 = conv3d_backward_simple(d_s1_conv, image, filters_2d)

    return dA, dB, dK1, db1, dK2, db2, dK3, db3

# Batch = (B, 32, 32, 3) | Label = (B)
def train_step(image_batch, label_batch, filters_2d, filters_3d, filters_3d_2, A, B, learning_rate=0.001):
    # Forward
    loss, probs, cache_dense, masks_pool1, masks_pool2, z1, z2, z3, s1, s2, s3, maxpooled_3d_images_batch, maxpooled_3d_images_batch_2 = forward_pass(image, filters_2d, filters_3d, filters_3d_2, A, B, label)

    # Backward
    dA, dB, dK1, db1, dK2, db2, dK3, db3 = backward_pass(image, cache_dense, masks_pool1, masks_pool2, z1, z2, z3, s1, s2, s3, maxpooled_3d_images_batch, maxpooled_3d_images_batch_2)

    # Update parameters
    filters_2d -= learning_rate * dK1
    filters_3d -= learning_rate * dK2
    filters_3d_2 -= learning_rate * dK3
    A -= learning_rate * dA
    B -= learning_rate * dB

    return loss

if __name__ == "__main__":
    # Initialisation des filtres et poids
    nf = 8
    filters_2d  = 0.01 * xp.random.randn(nf, 3, 3, 3).astype(xp.float32)
    filters_3d  = 0.01 * xp.random.randn(nf, 3, 3, nf).astype(xp.float32)
    filters_3d_2= 0.01 * xp.random.randn(nf, 3, 3, nf).astype(xp.float32)
    A           = 0.01 * xp.random.randn(nf*8*8, 10).astype(xp.float32)
    B           = xp.zeros(10, dtype=xp.float32)
    # Entraînement sur un batch d'images
    num_epochs = 1
    nb_img = 1000
    for epoch in range(num_epochs):
        total_loss = 0
        for i in tqdm(range(nb_img)):  # Limiter à 10 images pour l'exemple
            image = x_train[i].reshape(32, 32, 3)
            label = y_train[i]
            loss = train_step(image, label, filters_2d, filters_3d, filters_3d_2, A, B)
            total_loss += loss
