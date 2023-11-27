import functools
import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import torchvision.models as models

from collections import Counter, defaultdict
from sklearn.decomposition import PCA
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms

# hyperparameters
device        = 'cpu'
lr            = 7e-2
epochs        = 10
milestones    = [epochs//2]
gamma         = 0.5
batch_size    = 512
momentum      = 0
weight_decay  = 0
n_bins        = 10

# pruning parameters
N_prunes = 10
pruning_method = prune.L1Unstructured
prune_amount = 0.3
fine_tuning_epochs = 5


# deep network
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(12, 10)
        self.fc2 = nn.Linear(10, 7)
        self.fc3 = nn.Linear(7, 5)
        self.fc4 = nn.Linear(5, 4)
        self.fc5 = nn.Linear(4, 3)
        self.fc6 = nn.Linear(3, 3)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        h3 = torch.tanh(self.fc3(h2))
        h4 = torch.tanh(self.fc4(h3))
        h5 = torch.tanh(self.fc5(h4))
        h6 = F.log_softmax(self.fc6(h5), dim=1)
        return h1, h2, h3, h4, h5, h6

model = Model().to(device)

# get the list of modules and weights to prune
parameters_to_prune = [
    (module, "weight") for module in model.modules()
    if isinstance(module, nn.Linear)
]

def process_dataset(dataset):
    # keep only images of two digits
    digit1 = 0
    digit2 = 2
    digit3 = 8

    indices = (dataset.targets == digit1) | (dataset.targets == digit2) | (dataset.targets == digit2)
    dataset.data, dataset.targets = dataset.data[indices], dataset.targets[indices]

    dataset.targets[dataset.targets == digit1] = 0
    dataset.targets[dataset.targets == digit2] = 1
    dataset.targets[dataset.targets == digit3] = 3

    dataset = TensorDataset(dataset.data.reshape(-1, 28*28), dataset.targets)

    return dataset

transform = transforms.ToTensor()
train_dataset = process_dataset(datasets.MNIST('../data', download=True, train=True, transform=transform))
test_dataset = process_dataset(datasets.MNIST('../data', download=True, train=False, transform=transform))

# use PCA to reduce to 12 dimensions
pca = PCA(n_components=12)
pca.fit(train_dataset.tensors[0])
train_dataset.tensors = (torch.from_numpy(pca.transform(train_dataset.tensors[0])).float(), train_dataset.tensors[1])
test_dataset.tensors = (torch.from_numpy(pca.transform(test_dataset.tensors[0])).float(), test_dataset.tensors[1])

# dataset loaders
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

# loss function
loss_function = nn.NLLLoss()

# optimizer
optimizer = optim.SGD(model.parameters(),
                      lr=lr,
                      momentum=momentum,
                      weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


train_accuracy = []
test_accuracy = []

pruning_times = []

layers = len(list(model.children()))
n_bins = 10
x_bins = np.linspace(0, 1, n_bins + 1)
h_bins = np.linspace(-1, 1, n_bins + 1)
unit = 1./len(train_loader.dataset)

MI_XH = []
MI_YH = []
TC = []

def run_inference_and_digitize(model, images, labels, x_bins, h_bins):
    images = images.to(device) 
    h_list = model(images)

    x_ = np.digitize(images.cpu(), x_bins)
    y_ = labels.cpu()
    h_s = []

    for layer, h in enumerate(h_list):
        bins = h_bins
        if layer == len(h_list) - 1:
            h = torch.exp(h)
            bins = x_bins
        h_ = np.digitize(h.cpu().detach().numpy(), bins)
        h_s.append(h_)

    return x_, y_, h_s

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except:
        return a.item()

total_epoch = -1 # Running count of the total number of epochs accomplished among all pruning steps
for pruning_step in range(N_prunes):

    for epoch in range(epochs if pruning_step == 0 else fine_tuning_epochs):
        total_epoch += 1

        # estimate the total correlation
        model.eval()

        # Accumulate digitized versions of the image, labels, and layers
        digitized_images = []
        digitized_labels = []
        digitized_layers = []
        for images, labels in train_loader:
            x_, y_, h_s = run_inference_and_digitize(model, images, labels, x_bins, h_bins)
            digitized_images.append(x_)
            digitized_labels.append(y_)
            digitized_layers.append(h_s)

        # next, accumulate probability densities for X, Y, H, (X, H), and (Y, H) respectively.
        p_X, p_Y = Counter(), Counter()
        p_Hs, p_XHs, p_YHs = [defaultdict(Counter) for _ in range(3)]
        p_H_marginals = [defaultdict(list) for _ in range(layers)]
        for x_, y_, h_s in zip(digitized_images, digitized_labels, digitized_layers):
            for layer, h_ in enumerate(h_s):
                p_H, p_XH, p_YH = p_Hs[layer], p_XHs[layer], p_YHs[layer]
                    
                for x_i, h_i, y_i in zip(x_, h_, y_):
                    p_X[totuple(x_i)] += unit
                    p_H[totuple(h_i)] += unit
                    p_Y[totuple(y_i)] += unit
                    p_XH[totuple((x_i, h_i))] += unit
                    p_YH[totuple((y_i, h_i))] += unit

                    # accumulate the marginal densities for calculating total correlation
                    if not p_H_marginals[layer]:
                        p_H_marginals[layer] = [Counter() for _ in range(len(h_i))]
                    for j, h_ij in enumerate(h_i):
                        p_Hj = p_H_marginals[layer][j]
                        p_Hj[totuple(h_ij)] += unit

        # now that we have density estimates, we can estimate the mutual information
        mi_xh = torch.zeros(layers)
        mi_yh = torch.zeros(layers)
        tc = torch.zeros(layers)
        for x_, y_, h_s in zip(digitized_images, digitized_labels, digitized_layers):
            for layer, h_ in enumerate(h_s):
                p_H, p_XH, p_YH = p_Hs[layer], p_XHs[layer], p_YHs[layer]
                for x_i, h_i, y_i in zip(x_, h_, y_):
                    px = p_X[totuple(x_i)]
                    ph = p_H[totuple(h_i)]
                    py = p_Y[totuple(y_i)]
                    pxh = p_XH[totuple((x_i, h_i))]
                    pyh = p_YH[totuple((y_i, h_i))]
                    mi_xh[layer] += math.log(pxh / (px * ph))
                    mi_yh[layer] += math.log(pyh / (py * ph))

                    marginal_prod = functools.reduce(
                        lambda a, b: a * b, 
                        [p_H_marginals[layer][j][totuple(h_ij)] for j, h_ij in enumerate(h_i)], 
                        1
                    )
                    tc[layer] += math.log(ph / marginal_prod)

        MI_XH.append(mi_xh)
        MI_YH.append(mi_yh)
        TC.append(tc)
            # print(f"Calculated MI_XH as {MI_XH[epoch, layer]} and MI_YH as {MI_YH[epoch, layer]}")

        # train loop
        model.train()
        accuracy = 0
        N = 0

        for batch_idx, (images, labels) in enumerate(train_loader, start = 1):
            images, labels = images.to(device), labels.to(device)

            h1, h2, h3, h4, h5, h6 = model(images)
            loss = loss_function(h6, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted_labels = torch.argmax(h6, dim = 1)
            accuracy += torch.sum((predicted_labels == labels).float()).item()

            N += images.shape[0]

            # print(
            #     'Train\t\tEpoch: {} \t'
            #     'Batch {}/{} ({:.0f}%) \t'
            #     'Batch Loss: {:.6f} \t'
            #     'Batch Accuracy: {:.6f}'.format(
            #         epoch+1,
            #         batch_idx,
            #         len(train_loader),
            #         100. * batch_idx / len(train_loader),
            #         loss.item(),
            #         100. * accuracy/N))

        train_accuracy.append(100. * accuracy/N)
        scheduler.step()

        # test loop
        model.eval()
        accuracy = 0
        N = 0

        for batch_idx, (images, labels) in enumerate(test_loader, start = 1):
            images, labels = images.to(device), labels.to(device)

            h1, h2, h3, h4, h5, h6 = model(images)

            predicted_labels = torch.argmax(h6, dim = 1)
            accuracy += torch.sum((predicted_labels == labels).float()).item()
            N += images.shape[0]

        test_accuracy.append(100. * accuracy / N)
        print("Test Accuracy:", test_accuracy[-1])

    # If this is the last pruning iteration, just break instead of pruning further
    if pruning_step == N_prunes - 1:
        break

    prune.global_unstructured(
        parameters = parameters_to_prune,
        pruning_method = pruning_method,
        amount = prune_amount
    )
    pruning_times.append(total_epoch)

    total_num_pruned = sum(
        (module.get_buffer(name + "_mask") == 0).sum().item()
        for module, name in parameters_to_prune
    )
    total_num_params = sum(
        module.get_parameter(name + "_orig").numel()
        for module, name in parameters_to_prune
    )
    print(f"Number of pruned parameters: {total_num_pruned}/{total_num_params}")


# plot results
plt.figure()
plt.title('Accuracy Versus Epoch')
plt.plot(range(len(train_accuracy)), train_accuracy, label='Train')
plt.plot(range(len(test_accuracy)), test_accuracy, label='Test')

# draw lines whenever pruning occurs
for x in pruning_times:
    plt.axvline(x = x, color = "red", linestyle = "--")

plt.legend()
plt.show()

plt.figure()
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel('MI_X,H')
ax.set_ylabel('MI_Y,H')
title = ax.set_title('Information Plane')
cmap = plt.get_cmap('gnuplot', total_epoch)
colors = [cmap(i) for i in np.linspace(0, 1, total_epoch + 1)]
ax.plot(MI_XH, MI_YH, '-', color='gray')
for layer in range(layers):
    im = ax.scatter(
        [mi_xh[layer] for mi_xh in MI_XH],
        [mi_yh[layer] for mi_yh in MI_YH],
        c = colors
    )
plt.show()

plt.figure()
plt.title("Total Correlation")
for layer in range(layers):
    plt.plot(range(len(TC)), [TC[i][layer] for i in range(len(TC))], label = f"Total Correlation for Layer {layer}")
plt.legend()
plt.show()