##### LOAD MODULES
# path tools
import os

# data loader
from tensorflow.keras.datasets import cifar10

# machine learning tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# classification model
from sklearn.neural_network import MLPClassifier

# other modules
import numpy as np
import cv2

##### DEFINE LABELS
labels = [
    "airplane", 
    "automobile", 
    "bird", 
    "cat", 
    "deer", 
    "dog", 
    "frog", 
    "horse", 
    "ship", 
    "truck"] # in alphabetical order

##### BASIC FUNCTIONS
# PREPROCESS IMAGES
def preprocess(X_train, X_test):
    
    # Convert images to greyscale. 
    X_train_grey = np.array([
        cv2.cvtColor(
            image, 
            cv2.COLOR_BGR2GRAY) for image in X_train])

    X_test_grey = np.array([
        cv2.cvtColor(
            image, 
            cv2.COLOR_BGR2GRAY) for image in X_test])


    # Scale images. 
    X_train_scaled = (X_train_grey)/255.0
    X_test_scaled = (X_test_grey)/255.0


    # Extract each dimension and flatten from 3 into 2 dimensional arrays.
    nsamples, nx, ny = X_train_scaled.shape 
    X_train_dataset = X_train_scaled.reshape((
        nsamples,
        nx * ny))

    nsamples, nx, ny = X_test_scaled.shape 
    X_test_dataset = X_test_scaled.reshape((
        nsamples,
        nx * ny)) 

    return(
        X_train_dataset, 
        X_test_dataset)

# EVALUATE CLASSIFIER PERFORMANCE
def evaluate(y_test, y_pred, labels, type):

    # Create classification report.
    report = classification_report(
        y_test, 
        y_pred, 
        target_names = labels)

    # Define path for saving report.
    outpath = os.path.join(
        "out",
        type + ".txt")

    # Save report.
    with open(outpath, 'w') as f:
        f.write(report)

##### PIPELINE
def run_classifier(type):

    # Load data. 
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Use preprocessing function on data.
    X_train_dataset, X_test_dataset = preprocess(X_train, X_test)

    # Train classifier. 
    clf.fit(
        X_train_dataset, 
        y_train)

    # Make predictions.
    y_pred = clf.predict(
        X_test_dataset)

    # Evaluate classifier performance.
    evaluate(
        y_test, 
        y_pred, 
        labels, 
        type)

##### DEFINE CLASSIFIER 
clf = MLPClassifier(
    random_state = 42,
    hidden_layer_sizes = (64, 10),
    learning_rate = "adaptive",
    early_stopping = True,
    verbose = True,
    max_iter = 100)

##### MAIN
def main():

    # Run classification process.
    run_classifier("mlp")

if __name__ == "__main__":
    main()