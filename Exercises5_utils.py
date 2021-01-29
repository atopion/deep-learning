import os
import fastprogress
import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader


# Custom printing
from IPython.display import Markdown, display
def printmd(string):
    display(Markdown(string))

def get_device(cuda_preference=True):
    """Gets pytorch device object. If cuda_preference=True and 
        cuda is available on your system, returns a cuda device.
    
    Args:
        cuda_preference: bool, default True
            Set to true if you would like to get a cuda device
            
    Returns: pytorch device object
            Pytorch device
    """
    
    print('cuda available:', torch.cuda.is_available(), 
          '; cudnn available:', torch.backends.cudnn.is_available(),
          '; num devices:', torch.cuda.device_count())
    
    use_cuda = False if not cuda_preference else torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    device_name = torch.cuda.get_device_name(device) if use_cuda else 'cpu'
    print('Using device', device_name)
    return device

def imshow(img, mean, std, label, ax, cls_name):
    """Undo normalization using mean and standarddeviation and show image.

    Args:
        img (torch.Tensor): Image to show
        mean (np.array shape (3,)): Vector of means per channel used to
            normalize the dataset.
        std (np.array shape (3,)): Vector of standard deviations per channel 
            used to normalize the dataset.
        label (int): Label of the image (as int)
        ax (pyplot subplot): Subplot to display the image in
        cls_name (str): label of the image
    """
    ####################
    ## YOUR CODE HERE ##
    ####################

    # Convert the images
    npimg = torchvision.utils.make_grid(img).numpy()
    npimg = np.transpose(npimg, (1, 2, 0))

    ax.set_title("label: {}".format(cls_name))
    ax.axis("off")
    ax.imshow(npimg)



def train(dataloader, optimizer, model, loss_fn, device, master_bar):
    """Run one training epoch.

    Args:
        dataloader (DataLoader): Torch DataLoader object to load data
        optimizer: Torch optimizer object
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over for each
            epoch to draw batches and display training progress

    Returns:
        float, float: Mean loss of this epoch, fraction of correct predictions
            on training set (accuracy)
    """
    epoch_loss = []
    epoch_mse_loss, epoch_kld_loss = [], []

    for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
        optimizer.zero_grad()
        model.train()

        # Forward pass
        y_hat, mu, logvar = model(x.to(device))

        # Compute loss
        total_loss, mse_loss, kld_loss = loss_fn(y_hat, x.to(device), mu, logvar)

        # Backward pass
        total_loss.backward()
        optimizer.step()

        # For plotting the train loss, save it for each sample
        epoch_loss.append(total_loss.item())
        epoch_mse_loss.append(mse_loss.item())
        epoch_kld_loss.append(kld_loss.item())


    # Return the mean loss and the accuracy of this epoch
    return np.mean(epoch_loss), np.mean(epoch_mse_loss), np.mean(epoch_kld_loss)


def validate(dataloader, model, loss_fn, device, master_bar):
    """Compute loss, accuracy and confusion matrix on validation set.

    Args:
        dataloader (DataLoader): Torch DataLoader object to load data
        model (nn.Module): Torch model to train
        loss_fn: Torch loss function
        device (torch.device): Torch device to use for training
        master_bar (fastprogress.master_bar): Will be iterated over to draw 
            batches and show validation progress

    Returns:
        float, float, torch.Tensor shape (10,10): Mean loss on validation set, 
            fraction of correct predictions on validation set (accuracy)
    """
    epoch_loss = []
    epoch_mse_loss, epoch_kld_loss = [], []  

    model.eval()
    with torch.no_grad():
        for x, y in fastprogress.progress_bar(dataloader, parent=master_bar):
            # make a prediction on validation set

            y_hat, mu, logvar = model(x.to(device))

            # Compute loss
            total_loss, mse_loss, kld_loss = loss_fn(y_hat, x.to(device), mu, logvar)

            # For plotting the test loss, save it for each sample
            epoch_loss.append(total_loss.item())
            epoch_mse_loss.append(mse_loss.item())
            epoch_kld_loss.append(kld_loss.item())

    # Return the mean loss, the accuracy and the confusion matrix
    return np.mean(epoch_loss), np.mean(epoch_mse_loss), np.mean(epoch_kld_loss)



def run_training(model, optimizer, loss_function, device, num_epochs, 
                train_dataloader, test_dataloader, verbose=False):
    """Run model training.

    Args:
        model (nn.Module): Torch model to train
        optimizer: Torch optimizer object
        loss_fn: Torch loss function for training
        device (torch.device): Torch device to use for training
        num_epochs (int): Max. number of epochs to train
        train_dataloader (DataLoader): Torch DataLoader object to load the
            training data
        val_dataloader (DataLoader): Torch DataLoader object to load the
            validation data
        early_stopper (EarlyStopper, optional): If passed, model will be trained
            with early stopping. Defaults to None.
        verbose (bool, optional): Print information about model training. 
            Defaults to False.

    Returns:
        list, list, list, list, torch.Tensor shape (10,10): Return list of train
            losses, validation losses, train accuracies, validation accuracies
            per epoch and the confusion matrix evaluated in the last epoch.
    """
    start_time = time.time()
    master_bar = fastprogress.master_bar(range(num_epochs))
    train_losses, test_losses = [],[]
    train_mse_losses, test_mse_losses = [],[]
    train_kld_losses, test_kld_losses = [],[]

    for epoch in master_bar:
        # Train the model
        epoch_train_loss, epoch_train_mse_loss, epoch_train_kld_loss = train(train_dataloader, optimizer, model, 
                                                                             loss_function, device, master_bar)
        # Validate the model
        epoch_test_loss, epoch_test_mse_loss, epoch_test_kld_loss   = validate(test_dataloader, model,
                                                                               loss_function, device, master_bar)

        # Save loss and acc for plotting
        train_losses.append(epoch_train_loss)
        test_losses.append(epoch_test_loss)

        train_mse_losses.append(epoch_train_mse_loss)
        test_mse_losses.append(epoch_test_mse_loss)

        train_kld_losses.append(epoch_train_kld_loss)
        test_kld_losses.append(epoch_test_kld_loss)

        if verbose:
            master_bar.write(f'Train loss: {epoch_train_loss:.2f}, test loss: {epoch_test_loss:.2f}')
            
            
    time_elapsed = np.round(time.time() - start_time, 0).astype(int)
    print(f'Finished training after {time_elapsed} seconds.')
    return train_losses, test_losses, train_mse_losses, test_mse_losses, train_kld_losses, test_kld_losses


def plot(title, label, train_results, test_results, ax=None, yscale='linear', save_path=None, 
         extra_pt=None, extra_pt_label=None):
    """Plot learning curves.

    Args:
        title (str): Title of plot
        label (str): x-axis label
        train_results (list): Results vector of training of length of number
            of epochs trained. Could be loss or accuracy.
        val_results (list): Results vector of validation of length of number
            of epochs. Could be loss or accuracy.
        yscale (str, optional): Matplotlib.pyplot.yscale parameter. 
            Defaults to 'linear'.
        save_path (str, optional): If passed, figure will be saved at this path.
            Defaults to None.
        extra_pt (tuple, optional): Tuple of length 2, defining x and y coordinate
            of where an additional black dot will be plotted. Defaults to None.
        extra_pt_label (str, optional): Legend label of extra point. Defaults to None.
    """
    
    epoch_array = np.arange(len(train_results)) + 1
    train_label, test_label = "Training "+label.lower(), "Test "+label.lower()
    
    sns.set(style='ticks')

    #if ax is not None:
    #    plt = ax

    plt.plot(epoch_array, train_results, epoch_array, test_results, linestyle='dashed', marker='o')
    legend = ['Train results', 'Test results']
    
    if extra_pt:
        ####################
        ## YOUR CODE HERE ##
        ####################
        plt.scatter(extra_pt[0]+1, extra_pt[1], c='black', zorder=9999)

        # END OF YOUR CODE #
        
    plt.legend(legend)
    if ax is not None:
        plt.set_xlabel('Epoch')
        plt.set_ylabel(label)
        plt.set_yscale(yscale)
        sns.despine(trim=True, offset=5)
        plt.set_title(title, fontsize=15)
    else:
        plt.xlabel('Epoch')
        plt.ylabel(label)
        plt.yscale(yscale)
        sns.despine(trim=True, offset=5)
        plt.title(title, fontsize=15) 
    
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight')
    
    if ax is None:
        plt.show()


# Helper function to find and print the best training results
def printBestValues(test_losses, test_mse_losses, test_kld_losses):
    # Find overall best value: val_loss (lower is better)
    best_loss_value = 10e8
    best_loss_epoch = 0
    for idx in range(len(test_losses)):
      if test_losses[idx] < best_loss_value:
        best_loss_value = test_losses[idx]
        best_loss_epoch = idx
    
    print("Lowest total validation loss:     {} (in epoch {})".format(best_loss_value, best_loss_epoch +1))

    # Find MSE best value: val_loss (lower is better)
    best_mse_loss_value = 10e8
    best_mse_loss_epoch = 0
    for idx in range(len(test_mse_losses)):
      if test_mse_losses[idx] < best_mse_loss_value:
        best_mse_loss_value = test_mse_losses[idx]
        best_mse_loss_epoch = idx
    
    print("Lowest MSE validation loss:     {} (in epoch {})".format(best_mse_loss_value, best_mse_loss_epoch +1))

    # Find KLD best value: val_loss (lower is better)
    best_kld_loss_value = 10e8
    best_kld_loss_epoch = 0
    for idx in range(len(test_kld_losses)):
      if test_kld_losses[idx] < best_kld_loss_value:
        best_kld_loss_value = test_kld_losses[idx]
        best_kld_loss_epoch = idx
    
    print("Lowest KLD validation loss:     {} (in epoch {})".format(best_kld_loss_value, best_kld_loss_epoch +1))

    # Return the epochs with the best values
    return best_loss_epoch, best_mse_loss_epoch, best_kld_loss_epoch