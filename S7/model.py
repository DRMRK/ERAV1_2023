import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms
from torchsummary import summary
from tqdm import tqdm


from utils import GetCorrectPredCount


# Base model: model_0
# Target:  get the basic structure, a model that predicts something.
# Started with one conv block and then a fully connected layer,
# realized adding conv reduces number of parameters
# while increasing what the model learns. Kept adding conv blocks until close to 8000 parameters
# Only this in augmentation: transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),

# Results:
# Parameters: 7,226, best train accuracy: 89.39. best test accuracy: 89.03

# Analysis:
# Model is small, works.
# Train accuracy is higher than test, hint of over fitting


class model_0(nn.Module):
    def __init__(self):
        super(model_0, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )  # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )

        self.fc1 = nn.Sequential(nn.Linear(16, 10), nn.ReLU())

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


################################################################################

#  model: model_1
# Target:  Improve on the test accuracy
# Started with adding Batchnorm to one layer, saw improvement, so kept adding to other layers.

# Results:
# Parameters: 7,322, best train accuracy: 99.62. best test accuracy: 98.95


# Analysis:
# Model is small, overfits, needs regularization.


class model_1(nn.Module):
    def __init__(self):
        super(model_1, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )  # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(nn.Linear(16, 10), nn.ReLU())

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


################################################################################

# model: model_2
# Target:  Reduce over fitting and improve test accuracy
# Started with adding Dropout to one layer, saw improvement, so kept adding to other layers.

# Results:
# Parameters: 7,322, best train accuracy: 99.53. best test accuracy: 99.18


# Analysis:
# Reduced over fitting Need to increase model complexity


class model_2(nn.Module):
    def __init__(self):
        super(model_2, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.1),
        )  # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(nn.Linear(16, 10), nn.ReLU())

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


################################################################################

# model: model_3
# Target:  Improve accuracy,
# Added FC layers at the tail end

# Results:
# Parameters: 7,464, best train accuracy: 99.55. best test accuracy: 99.04

# Analysis:
# Did not work out, model_1 is still better. Try data augmentation with model_1


class model_3(nn.Module):
    def __init__(self):
        super(model_3, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.Dropout(p=0.1),
        )  # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(nn.Linear(16, 10), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(10, 10), nn.ReLU())

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)


################################################################################

# model: model_1 with augmentation
# Target:  Improve accuracy,

# Results:
# Parameters: 7,322, best train accuracy: 99.06. best test accuracy: 99.3

# Analysis:
# Improved test accuracy a little bit.
##################################################################################


class model_1_4(nn.Module):
    def __init__(self):
        super(model_1_4, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
        )  # output_size = 26

        self.convblock2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=18,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(18),
        )
        self.convblock3 = nn.Sequential(
            nn.Conv2d(
                in_channels=18,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(3, 3),
                padding=0,
                bias=False,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )

        self.fc1 = nn.Sequential(nn.Linear(16, 10), nn.ReLU())

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.convblock4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=-1)




# Train data transformations
def train_transforms():
    return transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.CenterCrop(22),
                ],
                p=0.2,
            ),
            # transforms.RandomAdjustSharpness(2, p=0.2),
            # transforms.RandomAffine(degrees=(-0.1, 0.2), translate=(0.1, 0.1)),
            transforms.RandomRotation((-10.0, 10.0), fill=0.1307),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


# Test data transformations
def test_transforms():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


def model_summary(model, input_size):
    summary(model, input_size)


def model_train(
    model,
    device,
    train_loader,
    optimizer,
    criterion,
    train_acc,
    train_losses,
):
    model.train()
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss += loss.item()

        # Back propagation
        loss.backward()
        optimizer.step()

        correct += GetCorrectPredCount(pred, target)
        processed += len(data)

        pbar.set_description(
            desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )
    train_acc.append(100 * correct / processed)
    train_losses.append(train_loss / len(train_loader))

    return train_acc, train_losses


def model_test(
    model,
    device,
    test_loader,
    criterion,
    test_acc,
    test_losses,
):
    model.eval()

    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(
                output, target, reduction="sum"
            ).item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target)

    test_loss /= len(test_loader.dataset)
    test_acc.append(100.0 * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_acc, test_losses
