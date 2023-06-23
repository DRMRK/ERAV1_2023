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
# Parameters: 7,226, best train accuracy: 99.31. best test accuracy: 99.00

# Analysis:
# Model is small, works.
# Train accuracy is higher than test over fitting


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
# Parameters: 7,322, best train accuracy: 99.7. best test accuracy: 99.09


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
# Parameters: 7,902, best train accuracy: 99.51. best test accuracy: 99.19

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


# class Net1(nn.Module):
#     # This defines the structure of the NN.
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
#         self.fc1 = nn.Linear(4096, 50)
#         self.fc2 = nn.Linear(50, 10)

#     def forward(self, x):
#         x = F.relu(self.conv1(x), 2)
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(self.conv3(x), 2)
#         x = F.relu(F.max_pool2d(self.conv4(x), 2))
#         x = x.view(-1, 4096)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return F.log_softmax(x, dim=1)


# dropout_value = 0.1


# class model_1(nn.Module):
#     def __init__(self):
#         super(model_1, self).__init__()
#         # Input Block
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 26

#         # CONVOLUTION BLOCK 1
#         self.convblock2 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=32,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(32),
#             nn.Dropout(dropout_value),
#         )  # output_size = 24

#         # TRANSITION BLOCK 1
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=32,
#                 out_channels=10,
#                 kernel_size=(1, 1),
#                 padding=0,
#                 bias=False,
#             ),
#         )  # output_size = 24
#         self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

#         # CONVOLUTION BLOCK 2
#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=10,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 10
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 8
#         self.convblock6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6
#         self.convblock7 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=1,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6

#         # OUTPUT BLOCK
#         self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=10,
#                 kernel_size=(1, 1),
#                 padding=0,
#                 bias=False,
#             ),
#             # nn.BatchNorm2d(10),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_value)
#         )

#         self.dropout = nn.Dropout(dropout_value)

#     def forward(self, x):
#         x = self.convblock1(x)
#         x = self.convblock2(x)
#         x = self.convblock3(x)
#         x = self.pool1(x)
#         x = self.convblock4(x)
#         x = self.convblock5(x)
#         x = self.convblock6(x)
#         x = self.convblock7(x)
#         x = self.gap(x)
#         x = self.convblock8(x)

#         x = x.view(-1, 10)
#         return F.log_softmax(x, dim=-1)


# class model_2(nn.Module):
#     def __init__(self):
#         super(model_2, self).__init__()
#         # Input Block
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 26

#         # # CONVOLUTION BLOCK 1
#         # self.convblock2 = nn.Sequential(
#         #     nn.Conv2d(
#         #         in_channels=16,
#         #         out_channels=32,
#         #         kernel_size=(3, 3),
#         #         padding=0,
#         #         bias=False,
#         #     ),
#         #     nn.ReLU(),
#         #     nn.BatchNorm2d(32),
#         #     nn.Dropout(dropout_value),
#         # )  # output_size = 24

#         # TRANSITION BLOCK 1
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=10,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#         )  # output_size = 24
#         self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

#         # CONVOLUTION BLOCK 2
#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=10,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 10
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 8
#         self.convblock6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6
#         self.convblock7 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=1,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6

#         # OUTPUT BLOCK
#         self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=10,
#                 kernel_size=(1, 1),
#                 padding=0,
#                 bias=False,
#             ),
#             # nn.BatchNorm2d(10),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_value)
#         )

#         self.dropout = nn.Dropout(dropout_value)

#     def forward(self, x):
#         x = self.convblock1(x)
#         # x = self.convblock2(x)
#         x = self.convblock3(x)
#         x = self.pool1(x)
#         x = self.convblock4(x)
#         x = self.convblock5(x)
#         x = self.convblock6(x)
#         x = self.convblock7(x)
#         x = self.gap(x)
#         x = self.convblock8(x)

#         x = x.view(-1, 10)
#         return F.log_softmax(x, dim=-1)


# class model_3(nn.Module):
#     def __init__(self):
#         super(model_3, self).__init__()
#         # Input Block
#         self.convblock1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=1,
#                 out_channels=12,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(12),
#             nn.Dropout(dropout_value),
#         )  # output_size = 26

#         # TRANSITION BLOCK 1
#         self.convblock3 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=12,
#                 out_channels=10,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#         )  # output_size = 24
#         self.pool1 = nn.MaxPool2d(2, 2)  # output_size = 12

#         # CONVOLUTION BLOCK 2
#         self.convblock4 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=10,
#                 out_channels=8,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(8),
#             nn.Dropout(dropout_value),
#         )  # output_size = 10
#         self.convblock5 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=8,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 8
#         self.convblock6 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=0,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6
#         self.convblock7 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=16,
#                 kernel_size=(3, 3),
#                 padding=1,
#                 bias=False,
#             ),
#             nn.ReLU(),
#             nn.BatchNorm2d(16),
#             nn.Dropout(dropout_value),
#         )  # output_size = 6

#         # OUTPUT BLOCK
#         self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=6))  # output_size = 1

#         self.convblock8 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels=16,
#                 out_channels=10,
#                 kernel_size=(1, 1),
#                 padding=0,
#                 bias=False,
#             ),
#             # nn.BatchNorm2d(10),
#             # nn.ReLU(),
#             # nn.Dropout(dropout_value)
#         )

#         self.dropout = nn.Dropout(dropout_value)

#     def forward(self, x):
#         x = self.convblock1(x)
#         # x = self.convblock2(x)
#         x = self.convblock3(x)
#         x = self.pool1(x)
#         x = self.convblock4(x)
#         x = self.convblock5(x)
#         x = self.convblock6(x)
#         x = self.convblock7(x)
#         x = self.gap(x)
#         x = self.convblock8(x)

#         x = x.view(-1, 10)
#         return F.log_softmax(x, dim=-1)


# Train data transformations
def train_transforms():
    return transforms.Compose(
        [
            #          transforms.RandomApply(
            #              [
            #                  transforms.CenterCrop(22),
            #              ],
            #              p=0.2,
            #          ),
            #transforms.RandomAdjustSharpness(2, p=0.2),
            #     transforms.RandomAffine(degrees=(-0.1, 0.2), translate=(0.1, 0.1)),
            #transforms.Resize((28, 28)),
            #transforms.RandomRotation((-5.0, 5.0), fill=0.1307),
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
