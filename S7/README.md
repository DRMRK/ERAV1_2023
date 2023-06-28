# ERAV1_2023 Assignment S7 QnA
> This repository contains example code for classifying the MNIST dataset using neural networks. The code is implemented in Python using the PyTorch framework.

### **Files**
- **model.py**: This files includes the implementation of a neural network model for the MNIST classification, configurations for augmenting the MNIST data and functions to show the data. All data specific files are kept here.

- **utils.py**: This files contains utility functions to train, test and show model performances.

- **S7-codes.ipynb**: This jupyter notebook provides an interactive environment for executing the code. It demonstrates the step-by-step process of loading the MNIST dataset, training the neural network models, and evaluating the classification performance and visualize the results.

### Experiments: 
> All detailed in model.py and S7-codes.ipynb

#### Exp1:
- Base model: model_0
- Target: get the basic structure, a model that predicts something.
Started with one conv block and then a fully connected layer,
realized adding conv reduces number of parameters
while increasing what the model learns. Kept adding conv blocks until close to 8000 parameters
Only this in augmentation: transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
#### Results:
- Parameters: 7,226, 
- best train accuracy: 89.39. 
- best test  accuracy: 89.03
#### Analysis:
- Model is small, works.
- Train accuracy is higher than test, hint of over fitting

#### Exp2:
- model: model_1
- Target: Improve on the test accuracy
Started with adding Batchnorm to one layer, saw improvement, so kept adding to other layers.
#### Results:
- Parameters: 7,322 
- best train accuracy: 99.62. 
- best test accuracy: 98.95
#### Analysis:
- Model is small, overfits, needs regularization.

#### Exp3:

- model: Model 2
- Target: Reduce over fitting and improve test accuracy
Started with adding Dropout to one layer, saw improvement, so kept adding to other layers.

#### Results:
- Parameters: 7,322, 
- best train accuracy: 99.49. 
- best test accuracy: 99.13
#### Analysis:
- Reduced over fitting Need to increase model complexity


### Exp3

- model: model_3
- Target: Improve accuracy,
- Added FC layers at the tail end to make model more complex
#### Results:
- Parameters: 7,464, 
- best train accuracy: 99.47.
- best test accuracy: 99.19
#### Analysis:
- Did not work out, model_1 is still better. Try data augmentation with model_1

### Exp4
- model_1 with data augmentation

#### Results
#### Results:
- Parameters: 7,322, 
- best train accuracy: 99.06.
- best test accuracy: last 5 epoch hovering 99.3

### Exp4

### **Requirements** 

To run the code in this repository, you need the dependencies included in the requirements.txt file.

You can install using pip 
````
pip install -r requirements.txt
````
To run the jupyter notebook follow these steps:

- Clone the repository
- Choose the environment in which you have installed all the dependencies.
- Execute the code in the jupyter notebook sequentially.
- Analyze the plots generated in the notebooks to gain insight into the training process and model performance. 
