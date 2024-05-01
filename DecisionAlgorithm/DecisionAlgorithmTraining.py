''' 
Title: Training of Classification Algorithms using scikit-learn python library
Coding Language: Python 

Author: Jason Thamel
Student ID: 2057941
University of Birmingham 

Description:
This script has been used to train and evaluated the performance of 5 classification algorithms 
'''

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import joblib
 
data = pd.read_csv('DC_Training_v3.csv')    # Load the training dataset from csv
data['Target'] = data['Target'].astype('category')  # Convert the numerical labels to categorical labels

# Get features and target variable
X = data.drop(columns=['Target'])   # Features
y = data['Target']                  # Target 

randomstate = 50 
xTraining, xValidation, yTraining, yValidation = train_test_split(X, y, test_size=0.2, random_state=randomstate)    # Split the data into training and validation sets

# Parameters for the algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": GaussianNB(),
    "SVC": SVC(kernel="linear", gamma=0.5, C=1.0, random_state=randomstate), #linear, rbf
    "MLP": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=randomstate)
}

# Training and evaluation
n_classes = len(y.cat.categories)  
classnames = y.unique()  # Define classnames 
results = {}


def algoAUC(model, xTraining, yTraining, xValidation, yValidation):
    # Ensure the model supports probability predictions
    if hasattr(model, 'predict_proba'):
        # Predict probabilities for training data
        y_trainProba = model.predict_proba(xTraining)
        # Predict probabilities for valid data
        y_validProba = model.predict_proba(xValidation)
        
        # Calculate AUC for training data
        aucTrain = roc_auc_score(yTraining, y_trainProba, multi_class="ovr", average="macro")
        # Calculate AUC for valid data
        aucValid = roc_auc_score(yValidation, y_validProba, multi_class="ovr", average="macro")
        
        print(f"AUC on Training Data: {aucTrain:.4f}")
        print(f"AUC on Validation Data: {aucValid:.4f}")
    else:
        print(" ")


def algoMetrics(model, xValidation, yValidation):
    # Make predictions
    yPredictions = model.predict(xValidation)

    # Calculate accuracy
    accuracy = accuracy_score(yValidation, yPredictions)
    print(f"Accuracy for {type(model).__name__}: {accuracy}")

    # Classification report
    print(f"\nClassification Report for {type(model).__name__}:\n")
    report = classification_report(yValidation, yPredictions, target_names=y.cat.categories, output_dict=True)
    print(classification_report(yValidation, yPredictions, target_names=y.cat.categories))

    recall = recall_score(yValidation, yPredictions, average='macro')
    precision = precision_score(yValidation, yPredictions, average='macro')
    missRate = 1 - recall

    print(f"\nThe Metrics for the {type(model).__name__} Algorithm are:\n")
    print(f"Recall: {recall}")
    print(f"Precision: {precision}")
    print(f"Miss Rate: {missRate}")
    
    # AUC for each class
    aucValues = results[type(model).__name__]["ROC AUC"]
    print("\nAUC for each class:")
    for i, class_name in enumerate(y.cat.categories):
        print(f"{class_name}: {aucValues[i]}")

    # Average AUC for all the classes
    print(f"\nMean AUC across all the classes: {np.mean(list(aucValues.values()))}\n")
    print("\n")

for name, model in models.items():
    
    # Train the model
    model.fit(xTraining, yTraining)

    # Save the algorithm models 
    #filename = f"C:/FYP/DecisionTree/models/v3_{name}_model_20.joblib"
    #joblib.dump(model, filename)
    
    # Make predictions
    yPredictions = model.predict(xValidation)
    
    # Calculate metrics
    accuracy = accuracy_score(yValidation, yPredictions)
    
    if hasattr(model, 'decision_function'):
        # This section is for the SVC Algorithm 
        y_score = model.decision_function(xValidation)
        # Calculate ROC curve and ROC area for each class
        falsePos = dict()
        truePos = dict()
        roc_auc = dict()
        for i in range(n_classes):
            falsePos[i], truePos[i], _ = roc_curve(pd.get_dummies(yValidation, columns=y.cat.categories).iloc[:, i], y_score[:, i])
            roc_auc[i] = auc(falsePos[i], truePos[i])
    else:
        # This section is for the Naive Bayes and MLP Algorithms
        y_predProba = model.predict_proba(xValidation)
        # Calculate ROC curve and ROC area for each class
        falsePos = dict()
        truePos = dict()
        roc_auc = dict()
        for i in range(n_classes):
            falsePos[i], truePos[i], _ = roc_curve(pd.get_dummies(yValidation, columns=y.cat.categories).iloc[:, i], y_predProba[:, i])
            roc_auc[i] = auc(falsePos[i], truePos[i])
    
    # Store the results
    results[type(model).__name__] = {
        "Accuracy": accuracy,
        "Confusion Matrix": confusion_matrix(yValidation, yPredictions),
        "False Positive Rate": falsePos,
        "True Positive Rate": truePos,
        "ROC AUC": roc_auc
    }

    # Print model name
    print(f"--- {name} ---")

    # Calculate and print accuracy, classification report, and metrics
    algoMetrics(model, xValidation, yValidation)
    print("\n")

    # Calculate and print AUC for both training and valid sets
    algoAUC(model, xTraining, yTraining, xValidation, yValidation)
    print("\n")

# Display confusion matrices using matplotlib
for name, metrics in results.items():

    graph = metrics["Confusion Matrix"]
    graphNormalise = graph.astype('float') / graph.sum(axis=1)[:, np.newaxis]           # Normalize the confusion matrix instead of numbers
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(graph, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    marks = range(len(y.cat.categories))
    plt.xticks(marks, y.cat.categories, rotation=45)
    plt.yticks(marks, y.cat.categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    for i in range(len(y.cat.categories)):
        for j in range(len(y.cat.categories)):
            plt.text(j, i, f"{graphNormalise[i, j]:.2f}", horizontalalignment="center", color="white" if graph[i, j] > graph.max() / 2 else "black")

    plt.show()

    
