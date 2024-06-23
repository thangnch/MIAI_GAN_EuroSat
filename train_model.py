
import g_model, d_model, data_loader
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 1024
num_epochs = 100
learning_rate_D = 1e-5
learning_rate_G = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:",device)

path_target = 'data/targets'
path_input = 'data/inputs'
test_ratio, train_ratio = 0.2, 0.8
rand_seed = 42
img_size = (64, 64)

# Count total data

num_test = int(len(os.listdir(path_target))*test_ratio)
num_train = int((int(len(os.listdir(path_target)))-num_test))

print("Number of train images:", num_train)
print("Number of test images:", num_test)

# Train , test split
random.seed(rand_seed)
train_indexs = np.array(random.sample(range(num_test+num_train), num_train))
mask = np.ones(num_train+num_test, dtype=bool)
mask[train_indexs] = False
#
# features = random.sample(os.listdir(path_input),num_test+num_train)
# targets = random.sample(os.listdir(path_target),num_test+num_train)
# random.Random(rand_seed).shuffle(features)
# random.Random(rand_seed).shuffle(targets)
features = ["IMG_{}.jpg".format(i) for i in range(1,27001)]
targets = ["IMG_{}.jpg".format(i) for i in range(1,27001)]
train_input_img_paths = np.array(features)[train_indexs]
train_target_img_path = np.array(targets)[train_indexs]
test_input_img_paths = np.array(features)[mask]
test_target_img_path = np.array(targets)[mask]

print("Ready to load")
# Set train, test loader
train_loader = data_loader.dataset(batch_size=batch_size, img_size=img_size, images_paths=train_input_img_paths, targets=train_target_img_path, path_input= path_input, path_target=path_target)
test_loader = data_loader.dataset(batch_size=batch_size, img_size=img_size, images_paths=test_input_img_paths, targets=test_target_img_path, path_input= path_input, path_target=path_target)


# Init model, optimizer and scheduler
discriminator = d_model.DModel().to(device)
generator = g_model.GModel().to(device)

bce = nn.BCEWithLogitsLoss()
l1loss = nn.L1Loss()

optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate_D)
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate_G)

scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=10, gamma=0.1)
scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=10, gamma=0.1)

best_generator_epoch_val_loss, best_discriminator_epoch_val_loss = np.inf, np.inf

print("Start training")
# Loop through N epochs
for epoch in range(num_epochs):
    print("#"*20, "Epoch = ", epoch)
    # Swicth to train mode
    discriminator.train()
    generator.train()

    discriminator_epoch_loss, generator_epoch_loss = 0, 0

    # Loop through all data
    for inputs, targets in train_loader:
        print(".")
        inputs, true_images = inputs.to(device), targets.to(device)

        ################ Train D Model
        optimizer_D.zero_grad()

        #  Make fake image from G_model
        fake_images = generator(inputs)#.detach()

        # Predict Fake/Real from D_model with fake image (hope will return 0) and count loss
        pred_fake = discriminator(fake_images).to(device)
        loss_fake = bce(pred_fake, torch.zeros(batch_size, device=device))

        # Predict Fake/Real from D_model with real image (hope will return 1) and count loss
        pred_real = discriminator(true_images).to(device)
        loss_real = bce(pred_real, torch.ones(batch_size, device=device))

        # Loss of D_model = avg 2 losses
        loss_D = (loss_fake + loss_real) / 2

        # backpropagation to caculate derivative
        loss_D.backward()
        optimizer_D.step()

        # Sum D_Model loss for this epoch
        discriminator_epoch_loss += loss_D.item()
        # all_loss_D.append(loss_D.item())

        ################ Train G Model
        optimizer_G.zero_grad()

        # Gen fake image from inputs
        fake_images = generator(inputs)#.detach()

        # Predict fake/real from D_model, caculate loss D_Model
        pred_fake = discriminator(fake_images).to(device)
        loss_G_bce = bce(pred_fake, torch.ones_like(pred_fake, device=device))

        # Caculate L1 loss, MAE between fake image and targets
        loss_G_l1 = l1loss(fake_images, true_images) * 100

        # Sum losses
        loss_G = loss_G_bce + loss_G_l1

        # Backpropagation to caculate derivative
        loss_G.backward()
        optimizer_G.step()

        # Sum GLoss
        generator_epoch_loss += loss_G.item()
        # all_loss_G.append(loss_G.item())

    # Caculate D and G loss for this epoch
    discriminator_epoch_loss /= len(train_loader)
    generator_epoch_loss /= len(train_loader)

    print("#"*20, "Start eval")
    # Switch to eval model
    discriminator.eval()
    generator.eval()

    discriminator_epoch_val_loss, generator_epoch_val_loss = 0, 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            fake_images = generator(inputs).detach()
            pred_fake = discriminator(fake_images).to(device)

            loss_G_bce = bce(pred_fake, torch.ones_like(pred_fake, device=device))
            loss_G_l1 = l1loss(fake_images, targets) * 100
            loss_G = loss_G_bce + loss_G_l1
            loss_D = bce(pred_fake.to(device), torch.zeros(batch_size, device=device))

            discriminator_epoch_val_loss += loss_D.item()
            generator_epoch_val_loss += loss_G.item()

    discriminator_epoch_val_loss /= len(test_loader)
    generator_epoch_val_loss /= len(test_loader)

    print(
        f"------Epoch [{epoch + 1}/{num_epochs}]------\nTrain Loss D: {discriminator_epoch_loss:.4f}, Val Loss D: {discriminator_epoch_val_loss:.4f}")
    print(f'Train Loss G: {generator_epoch_loss:.4f}, Val Loss G: {generator_epoch_val_loss:.4f}')

    # Save best weight
    if discriminator_epoch_val_loss < best_discriminator_epoch_val_loss:
        # discriminator_epoch_val_loss = best_discriminator_epoch_val_loss
        best_discriminator_epoch_val_loss = discriminator_epoch_val_loss
        torch.save(discriminator.state_dict(), "models/discriminator.pth")
        print("Save D at epoch ", epoch)
    if generator_epoch_val_loss < best_generator_epoch_val_loss:
        # generator_epoch_val_loss = best_generator_epoch_val_loss
        best_generator_epoch_val_loss = generator_epoch_val_loss
        torch.save(generator.state_dict(), "models/generator.pth")
        print("Save G at epoch ", epoch)
    # scheduler_D.step()
    # scheduler_G.step()
