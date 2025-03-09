import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

# Load the train and test sets
X_train = pd.read_csv('data/X_train.csv')
X_test = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv')
y_test = pd.read_csv('data/y_test.csv')

# Convert y_train and y_test to 1D arrays
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# Function to evaluate model performance
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return accuracy, precision, recall, f1, roc_auc

# Baseline model: Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg_results = evaluate_model(log_reg, X_train, y_train, X_test, y_test)

# Baseline model: Decision Tree
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree_results = evaluate_model(decision_tree, X_train, y_train, X_test, y_test)

# Print results
print("Logistic Regression Results:")
print(f"Accuracy: {log_reg_results[0]:.4f}")
print(f"Precision: {log_reg_results[1]:.4f}")
print(f"Recall: {log_reg_results[2]:.4f}")
print(f"F1 Score: {log_reg_results[3]:.4f}")
print(f"ROC-AUC: {log_reg_results[4]:.4f}")

print("\nDecision Tree Results:")
print(f"Accuracy: {decision_tree_results[0]:.4f}")
print(f"Precision: {decision_tree_results[1]:.4f}")
print(f"Recall: {decision_tree_results[2]:.4f}")
print(f"F1 Score: {decision_tree_results[3]:.4f}")
print(f"ROC-AUC: {decision_tree_results[4]:.4f}")

# Cross-validation for Logistic Regression
log_reg_cv_scores = cross_val_score(log_reg, X_train, y_train, cv=5, scoring='accuracy')
print("\nLogistic Regression Cross-Validation Accuracy Scores:", log_reg_cv_scores)
print("Mean Cross-Validation Accuracy:", log_reg_cv_scores.mean())

# Cross-validation for Decision Tree
decision_tree_cv_scores = cross_val_score(decision_tree, X_train, y_train, cv=5, scoring='accuracy')
print("\nDecision Tree Cross-Validation Accuracy Scores:", decision_tree_cv_scores)
print("Mean Cross-Validation Accuracy:", decision_tree_cv_scores.mean())

# Hyperparameter tuning for Decision Tree
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=decision_tree, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("\nBest Hyperparameters for Decision Tree:", grid_search.best_params_)
best_decision_tree = grid_search.best_estimator_

# Evaluate the best Decision Tree model
best_decision_tree_results = evaluate_model(best_decision_tree, X_train, y_train, X_test, y_test)

print("\nBest Decision Tree Results:")
print(f"Accuracy: {best_decision_tree_results[0]:.4f}")
print(f"Precision: {best_decision_tree_results[1]:.4f}")
print(f"Recall: {best_decision_tree_results[2]:.4f}")
print(f"F1 Score: {best_decision_tree_results[3]:.4f}")
print(f"ROC-AUC: {best_decision_tree_results[4]:.4f}")

# Feature importance for Decision Tree
feature_importances = pd.DataFrame(best_decision_tree.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)

print("\nFeature Importances for Decision Tree:")
print(feature_importances)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances.importance, y=feature_importances.index)
plt.title('Feature Importances for Decision Tree')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()