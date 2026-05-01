from cifar_conv_train_gpu_v2 import evaluate_model, load_model, load_cifar10

x_train, y_train, x_test, y_test = load_cifar10()
filters_2d, filters_3d, filters_3d_2, A, B_bias = load_model("params_cifar_conv_gpu.npz")
accuracy, avg_loss = evaluate_model(x_test, y_test, filters_2d, filters_3d, filters_3d_2, A, B_bias, batch_size=256, max_samples=5000)
print("accuracy :", accuracy, "| avg_loss :", avg_loss)