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
from sklearn.model_selection import train_valid_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import joblib

# Load the training dataset
data = pd.read_csv('DC_Training_v3.csv') 

# Convert the numerical labels to categorical labels
data['Target'] = data['Target'].astype('category')

# Prepare features and target variable
X = data.drop(columns=['Target'])  # Features
y = data['Target']  # Target variable

# Set the random state for all the models
randomstate = 50 

# Split the data into training and validation sets
Xtraining, Xvalidation, yTraining, yValidation = train_valid_split(X, y, valid_size=0.2, random_state=randomstate)

# Set the parameters for each Algorithm
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

# Training and evaluation
results = {}
for name, model in models.items():
    
    # Train the model
    model.fit(Xtraining, yTraining)

    # Save the model
    filename = f"C:/FYP/DecisionTree/models/v3_{name}_model_20.joblib"
    joblib.dump(model, filename)
    
    # Make predictions
    yPredictions = model.predict(Xvalidation)
    
    # Calculate metrics
    accuracy = accuracy_score(yValidation, yPredictions)
    
    if hasattr(model, 'decision_function'):
        # Compute decision function scores for SVC
        y_score = model.decision_function(Xvalidation)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(pd.get_dummies(yValidation, columns=y.cat.categories).iloc[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    else:
        # Compute probabilities for Naive Bayes and MLP
        y_predProba = model.predict_proba(Xvalidation)
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(pd.get_dummies(yValidation, columns=y.cat.categories).iloc[:, i], y_predProba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Store the results
    results[type(model).__name__] = {
        "Accuracy": accuracy,
        "Confusion Matrix": confusion_matrix(yValidation, yPredictions),
        "FPR": fpr,
        "TPR": tpr,
        "ROC AUC": roc_auc
    }

# Display confusion matrices using matplotlib
for name, metrics in results.items():

    cm = metrics["Confusion Matrix"]
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"{name} Confusion Matrix")
    plt.colorbar()
    tick_marks = range(len(y.cat.categories))
    plt.xticks(tick_marks, y.cat.categories, rotation=45)
    plt.yticks(tick_marks, y.cat.categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    for i in range(len(y.cat.categories)):
        for j in range(len(y.cat.categories)):
            plt.text(j, i, str(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.show()


def algoAUC(model, Xtraining, yTraining, Xvalidation, yValidation):
    # Ensure the model supports probability predictions
    if hasattr(model, 'predict_proba'):
        # Predict probabilities for training data
        y_trainProba = model.predict_proba(Xtraining)
        # Predict probabilities for valid data
        y_validProba = model.predict_proba(Xvalidation)
        
        # Calculate AUC for training data
        aucTrain = roc_auc_score(yTraining, y_trainProba, multi_class="ovr", average="macro")
        # Calculate AUC for valid data
        aucValid = roc_auc_score(yValidation, y_validProba, multi_class="ovr", average="macro")
        
        print(f"AUC on Training Data: {aucTrain:.4f}")
        print(f"AUC on Validation Data: {aucValid:.4f}")
    else:
        print("Error")


def algoMetrics(model, Xvalidation, yValidation):
    # Make predictions
    yPredictions = model.predict(Xvalidation)

    # Calculate accuracy
    accuracy = accuracy_score(yValidation, yPredictions)
    print(f"Accuracy for {type(model).__name__}: {accuracy}")

    # Classification report
    print(f"\nClassification Report for {type(model).__name__}:\n")
    report = classification_report(yValidation, yPredictions, target_names=y.cat.categories, output_dict=True)
    print(classification_report(yValidation, yPredictions, target_names=y.cat.categories))

    # Recall (True Positive Rate, TPR)
    sensitivity = recall_score(yValidation, yPredictions, average='macro')

    # Positive Predictive Value (Precision, PPV)
    precision = precision_score(yValidation, yPredictions, average='macro')

    # Miss Rate (False Negative Rate, FNR)
    miss_rate = 1 - sensitivity

    print(f"\nMetrics for {type(model).__name__}:\n")
    print(f"Recall (TPR): {sensitivity}")
    print(f"Positive Predictive Value (PPV): {precision}")
    print(f"Miss Rate (FNR): {miss_rate}")
    
    # AUC for each class
    aucValues = results[type(model).__name__]["ROC AUC"]
    print("\nAUC for each class:")
    for i, class_name in enumerate(y.cat.categories):
        print(f"{class_name}: {aucValues[i]}")
    print(f"\nMean AUC across all classes: {np.mean(list(aucValues.values()))}\n")
    print("\n")

# Train and evaluate models
for name, model in models.items():
    # Train the model
    model.fit(Xtraining, yTraining)

    # Print model name
    print(f"--- {name} ---")

    # Calculate and print accuracy, classification report, and metrics
    algoMetrics(model, Xvalidation, yValidation)
    print("\n")

    # Calculate and print AUC for both training and valid sets
    algoAUC(model, Xtraining, yTraining, Xvalidation, yValidation)
    print("\n")
