# ==== MACHINE LEARNING PHASES OF THE ISING MODEL ====

# ==== IMPORTS AND CONFIGURATION ====
# In[]:
import torch
import numpy as np
import matplotlib as mpl 
import matplotlib.pylab as plt

plt.rcParams['legend.fontsize'] = 15
plt.rcParams['xtick.labelsize'] = 15 
plt.rcParams['ytick.labelsize'] = 15 
plt.rcParams['axes.labelsize'] = 20 
plt.rcParams['axes.titlesize'] = 19 
plt.rc('font', **{'family': 'serif', 'serif':['Computer Modern']})
plt.rc('text', usetex=True)


# ==== LOAD DATASETS ====
# In[]:
# Critical temperatures of square and triangular lattices in thermodynamic limit
Tc_train = 2 / np.log(1 + np.sqrt(2))
Tc_test = 4 / np.log(3)

# Load training sets - Square lattices for different sizes, L
lattices_train = [
    torch.load('data/square/L10/lattices.pt', weights_only=False),
    torch.load('data/square/L20/lattices.pt', weights_only=False),
    torch.load('data/square/L30/lattices.pt', weights_only=False),
    torch.load('data/square/L40/lattices.pt', weights_only=False),
    torch.load('data/square/L60/lattices.pt', weights_only=False),
]
temperatures_train = torch.stack([
    torch.load('data/square/L10/temperatures.pt', weights_only=False),
    torch.load('data/square/L20/temperatures.pt', weights_only=False),
    torch.load('data/square/L30/temperatures.pt', weights_only=False),
    torch.load('data/square/L40/temperatures.pt', weights_only=False),
    torch.load('data/square/L60/temperatures.pt', weights_only=False)
])

labels_train = torch.stack([
    (temps < Tc_train).int() for temps in temperatures_train
])

# Load test sets - Triangular lattices for different sizes, L
lattices_test = [
    torch.load('data/triangular/L10/lattices.pt', weights_only=False),
    torch.load('data/triangular/L20/lattices.pt', weights_only=False),
    torch.load('data/triangular/L30/lattices.pt', weights_only=False),
    torch.load('data/triangular/L40/lattices.pt', weights_only=False),
    torch.load('data/triangular/L60/lattices.pt', weights_only=False)
]
temperatures_test = torch.stack([
    torch.load('data/triangular/L10/temperatures.pt', weights_only=False),
    torch.load('data/triangular/L20/temperatures.pt', weights_only=False),
    torch.load('data/triangular/L30/temperatures.pt', weights_only=False),
    torch.load('data/triangular/L40/temperatures.pt', weights_only=False),
    torch.load('data/triangular/L60/temperatures.pt', weights_only=False)
])

labels_test = torch.stack([
    (temps < Tc_test).int() for temps in temperatures_test
])


# ==== DEFINE MODEL ====
# In[]:
class CNN(torch.nn.Module):
    """ 
    Convolutional Neural Network from Carrasquilla, J., Melko, R. Machine learning phases of matter. 
    Nature Phys 13, 431–434 (2017).

    :param L: Side length of Ising lattice
    """
    def __init__(self, L):
        super().__init__() 
        dims_conv = (L - 2) // 1 + 1

        self._conv = torch.nn.Conv2d(kernel_size=2, stride=1, in_channels=1, out_channels=1)
        self._relu = torch.nn.ReLU()
        self._flatten = torch.nn.Flatten()
        self._linear1 = torch.nn.Linear(in_features=dims_conv ** 2, out_features=64)
        self._dropout = torch.nn.Dropout(p=0.5)
        self._linear2 = torch.nn.Linear(in_features=64, out_features=2)

        self._network = torch.nn.Sequential(
            self._conv,
            self._relu,
            self._flatten,
            self._linear1,
            self._relu,
            self._dropout,
            self._linear2,
        )
    def forward(self, x):
        x = x.unsqueeze(1)
        return self._network(x)

# Create model for each L
lengths_list = [10, 20, 30, 40, 60]
models = [CNN(L) for L in lengths_list]


# ==== TRAINING ====
# In[]:

lr = 1e-5
epochs = 150
batch_size = 5
loss_fn = torch.nn.CrossEntropyLoss()

model_losses = []
model_accuracies_train, model_accuracies_test = [], []

for i, model in enumerate(models):

    # Select data for each L 
    lattices_train

    # Shuffle lattices 
    perm = torch.randperm(lattices_train[i].size(0))
    X_train = lattices_train[i][perm]
    y_train = labels_train[i][perm]

    # Instantiate Adam optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)

    losses = []
    accuracies_train, accuracies_test = [], []

    for epoch in range(epochs):
        model.train()

        for batch in range(len(lattices_train[i]) // batch_size):
            # Batch set
            start_idx = batch * batch_size
            X = X_train[start_idx : start_idx + batch_size]
            y = y_train[start_idx : start_idx + batch_size].long()

            # Calculate output & loss
            output = model(X)
            loss = loss_fn(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        with torch.no_grad():
            model.eval()

            # Evaluate accuracies over training and test sets
            train_output = model(lattices_train[i])
            test_output = model(lattices_test[i])

            predictions_train = torch.argmax(train_output, dim=1)
            predictions_test = torch.argmax(test_output, dim=1)

            accuracy_train = (predictions_train == labels_train[i]).to(int).sum() / labels_train[i].size(0)
            accuracy_test = (predictions_test == labels_test[i]).to(int).sum() / labels_test[i].size(0)

            accuracies_train.append(accuracy_train) 
            accuracies_test.append(accuracy_test)

    model_losses.append(losses)
    model_accuracies_train.append(accuracies_train)
    model_accuracies_test.append(accuracies_test)


# ==== PLOTTING ====
# In[]: Plot first the training and test curves.

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot on the first subplots
axes[0].set_title("Training curves")
axes[1].set_title("Test curves")
axes[0].set_xlabel('Epoch')
axes[1].set_xlabel('Epoch')
axes[0].set_xlim(0, epochs - 1)
axes[1].set_xlim(0, epochs - 1)
axes[0].set_ylim(0.45, 1.03)
axes[1].set_ylim(0.45, 1.03)

for i, train_curve in enumerate(model_accuracies_train):
    axes[0].plot(train_curve, label=f'$L = {lengths_list[i]}$', linewidth=1.7, markersize=4)
    axes[1].plot(model_accuracies_test[i], linewidth=1.7, markersize=4)

plt.tight_layout()
axes[0].legend()
plt.show()

# In[]: Calculate and plot the average output for each temperature for the training set and the test set.

# Evalute output for entire training and test sets.
outputs_train = torch.stack([
    models[i](lattices_train[i]) for i in range(len(lengths_list))
])
outputs_test = torch.stack([
    models[i](lattices_test[i]) for i in range(len(lengths_list))
])
predictions_train = torch.argmax(outputs_train, dim=2)
predictions_test = torch.argmax(outputs_test, dim=2)

# Remove duplicates from temperature data and count the number of datapoints per temperature
unique_temp_test, counts = torch.unique(temperatures_test[0], return_counts=True)
num_copies_T_test = counts[0].item()
unique_temp_train, counts = torch.unique(temperatures_train[0], return_counts=True)
num_copies_T_train = counts[0].item()

# Sort the predictions by temperature
sort_by_temp_test = torch.argsort(temperatures_test, dim=1)
sort_by_temp_train = torch.argsort(temperatures_train, dim=1)

sorted_pred_test = torch.gather(predictions_test, dim=1, index=sort_by_temp_test)
sorted_pred_train = torch.gather(predictions_train, dim=1, index=sort_by_temp_train)

avg_pred_test = sorted_pred_test.view(sorted_pred_test.shape[0], -1, num_copies_T_test).sum(dim=2) / num_copies_T_test
avg_pred_train = sorted_pred_train.view(sorted_pred_train.shape[0], -1, num_copies_T_train).sum(dim=2) / num_copies_T_train

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
axes[0].set_title("Square lattices")
axes[1].set_title("Triangular lattices")
axes[0].set_xlabel('Temperature')
axes[1].set_xlabel('Temperature')
axes[0].set_ylabel('Accuracy')

for i, train_curve in enumerate(avg_pred_test):
    axes[0].plot(unique_temp_train, avg_pred_train[i], label=f'$L = {lengths_list[i]}$', linewidth=1.1, marker='o', markersize=4)
    axes[1].plot(unique_temp_test, avg_pred_test[i], linewidth=1.1, marker='o', markersize=4)

axes[0].axvline(x=Tc_train, color='black', linestyle='dashed')
axes[1].axvline(x=Tc_test, color='black', linestyle='dashed')
plt.tight_layout()
axes[0].legend()
plt.show()


# In[]: Plot the average accuracy with respect to each temperature

# Sort the labels and predictions by temperature
sorted_labels_test = torch.gather(labels_test, dim=1, index=sort_by_temp_test)

# Take average for each temperature
accuracies_test = (sorted_pred_test == sorted_labels_test)
accuracies_test = accuracies_test.view(accuracies_test.shape[0], -1, num_copies_T_test).sum(dim=2) / num_copies_T_test

# Sort the labels and predictions by temperature
sorted_labels_train = torch.gather(labels_train, dim=1, index=sort_by_temp_train)

# Take average for each temperature
accuracies_train = (sorted_pred_train == sorted_labels_train)
accuracies_train = accuracies_train.view(accuracies_train.shape[0], -1, num_copies_T_train).sum(dim=2) / num_copies_T_train

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

# Plot on the first subplots
axes[0].set_title("Square lattices")
axes[1].set_title("Triangular lattices")
axes[0].set_xlabel('Temperature')
axes[1].set_xlabel('Temperature')
axes[0].set_ylabel('Accuracy')

for i, train_curve in enumerate(avg_pred_test):
    axes[0].plot(unique_temp_train, accuracies_train[i], label=f'$L = {lengths_list[i]}$', linewidth=1.1, marker='o', markersize=4)
    axes[1].plot(unique_temp_test, accuracies_test[i], linewidth=1.1, marker='o', markersize=4)

axes[0].axvline(x=Tc_train, color='black', linestyle='dashed')
axes[1].axvline(x=Tc_test, color='black', linestyle='dashed')

plt.tight_layout()
axes[0].legend()
plt.show()
