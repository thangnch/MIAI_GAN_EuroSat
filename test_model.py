import random
import numpy as np
import os
import data_loader, g_model
import torch
import matplotlib.pyplot as plt


path_target = 'data/targets'
path_input = 'data/inputs'
test_ratio, train_ratio = 0.2, 0.8
rand_seed = 42
img_size = (64, 64)

# Count total data

num_test = int(len(os.listdir(path_target))*test_ratio)
num_train = int((int(len(os.listdir(path_target)))-num_test))

print("Number of train samples:", num_train)
print("Number of test samples:", num_test)

# Train , test split
random.seed(rand_seed)
train_idxs = np.array(random.sample(range(num_test+num_train), num_train))
mask = np.ones(num_train+num_test, dtype=bool)
mask[train_idxs] = False
#
# images = {}
# features = random.sample(os.listdir(path_input),num_test+num_train)
# targets = random.sample(os.listdir(path_target),num_test+num_train)
# random.Random(rand_seed).shuffle(features)
# random.Random(rand_seed).shuffle(targets)
features = ["IMG_{}.jpg".format(i) for i in range(1,27001)]
targets = ["IMG_{}.jpg".format(i) for i in range(1,27001)]
train_input_img_paths = np.array(features)[train_idxs]
train_target_img_path = np.array(targets)[train_idxs]
test_input_img_paths = np.array(features)[mask]
test_target_img_path = np.array(targets)[mask]
# Test after train
random.Random(rand_seed).shuffle(test_target_img_path)
random.Random(rand_seed).shuffle(test_input_img_paths)
subset_loader = data_loader.dataset(batch_size=5, img_size=img_size, images_paths=test_input_img_paths,
                        targets=test_target_img_path)

generator = g_model.GModel()
generator.load_state_dict(torch.load('models/generator.pth'))
generator.eval()
for X, y in subset_loader:
    fig, axes = plt.subplots(5, 3, figsize=(9, 9))

    for i in range(5):
        axes[i, 0].imshow(np.transpose(X.numpy()[i], (1, 2, 0)))
        axes[i, 0].set_title("Input")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.transpose(y.numpy()[i], (1, 2, 0)))
        axes[i, 1].set_title("Target")
        axes[i, 1].axis('off')

        generated_image = generator(X[i].unsqueeze(0)).detach().numpy()[0]
        axes[i, 2].imshow(np.transpose(generated_image, (1, 2, 0)))
        axes[i, 2].set_title("Generated")
        axes[i, 2].axis('off')

    # Adjust layout
    plt.tight_layout()
    plt.savefig('Test.jpg')
    plt.show()
    break
