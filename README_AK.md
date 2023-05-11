# Assignment 2 - Classification on the Cifar10 dataset
This assignment is ***Part 2*** of the portfolio exam for ***Visual Analytics S23***. The exam consists of 4 assignments in total (3 class assignments and 1 self-assigned project).

## 1.1. Contribution
The initial assignment was created partially in collaboration with other students in the course, also making use of code provided as part of the course. The final code is my own. Several adjustments have been made since the initial hand-in.

Here is the link to the GitHub repository containing the code for this assignment: ADD

## 1.2. Assignment description by Ross
*(**NB!** This description has been edited for brevity. Find the full instructions in ```README_rdkm.md```.)*

For this assignment, we'll be writing scripts which classify the ```Cifar10``` dataset. You should write code which does the following:

- Load the Cifar10 dataset
- Preprocess the data (e.g. greyscale, reshape)
- Train a classifier on the data
- Save a classification report

You should write one script which does this for a logistic regression classifier **and** one which does it for a neural network classifier. In both cases, you should use the machine learning tools available via ```scikit-learn```.

## 1.3. Methods
The purpose of this script is to train two different types of classifiers on the Cifar10 machine learning dataset and evaluate their performances. First, the Cifar10 dataset is loaded into the script in separate train and test sets. The images are preprocessed by being converted to greyscale, scaled, and flattened into 2-dimensional space. Two types of classifier can then be fitted to the data: one logistic regression classifier and one neural net classifier, depending on which script the user chooses to run. Finally, predictions are made on the test dataset and the model's performance is evaluated and the classification report is saved.

## 1.4. Usage
### 1.4.1. Prerequisites
This code was written and executed in the UCloud application's Coder Python interface (version 1.77.3, running Python version 3.9.2). UCloud provides virtual machines with a Linux-based operating system, therefore, the code has been optimized for Linux and may need adjustment for Windows and Mac.
