import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class CNN(nn.Module):
    """Convolutional Neural Network.
    
    We provide a simple network with a Conv layer, followed by pooling,
    and a fully connected layer. Modify this to test different architectures,
    and hyperparameters, i.e. different number of layers, kernel size, feature
    dimensions etc.

    See https://pytorch.org/docs/stable/nn.html for a list of different layers
    in PyTorch.
    """

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 6)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """Forward pass of network."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

    def write_weights(self, fname):
        """ Store learned weights of CNN """
        torch.save(self.state_dict(), fname)

    def load_weights(self, fname):
        """
        Load weights from file in fname.
        The evaluation server will look for a file called checkpoint.pt
        """
        ckpt = torch.load(fname)
        self.load_state_dict(ckpt)


def get_loss_function():
    """Return the loss function to use during training. We use
       the Cross-Entropy loss for now.
    
    See https://pytorch.org/docs/stable/nn.html#loss-functions.
    """
    return nn.CrossEntropyLoss()


def get_optimizer(network, lr=0.001, momentum=0.9):
    """Return the optimizer to use during training.
    
    network specifies the PyTorch model.

    See https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer.
    """

    # The fist parameter here specifies the list of parameters that are
    # learnt. In our case, we want to learn all the parameters in the network
    return optim.SGD(network.parameters(), lr=lr, momentum=momentum)
