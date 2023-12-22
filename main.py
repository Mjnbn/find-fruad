import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv('creditcard.csv')

X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]  # Target variable

# Step 3: Clustering - K-Means
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X)
cluster_labels = kmeans.labels_

# Get the number of fraud positive cases in each cluster
fraud_positive_cases = y[y == 1]
fraud_positive_clusters = cluster_labels[y == 1]
fraud_positive_count_in_clusters = pd.Series(fraud_positive_clusters).value_counts()

# Calculate the total number of fraud cases
total_fraud_cases = len(fraud_positive_cases)

# Calculate the percentage of fraud in each cluster
fraud_percentage_in_clusters = (fraud_positive_count_in_clusters / total_fraud_cases) * 100

print("K-Means Clustering Results:")
print("Cluster Centers:")
print(kmeans.cluster_centers_)
print("\n")

print("Number of fraud positive cases in each cluster:")
print(fraud_positive_count_in_clusters)
print("\n")

# Print the percentage of fraud in each cluster
print("Percentage of fraud in each cluster:")
print(fraud_percentage_in_clusters)

x = data.drop('Class', axis=1)
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

smote = SMOTE(sampling_strategy=1, random_state=42)
X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)

under_sampler = RandomUnderSampler(sampling_strategy=1, random_state=42)
X_train_undersampled, y_train_undersampled = under_sampler.fit_resample(X_train_oversampled, y_train_oversampled)

scaler = StandardScaler()
X_train_undersampled = scaler.fit_transform(X_train_undersampled)
X_test = scaler.transform(X_test)

# SVM

classifier = SVC(kernel='poly', C=0.1, gamma=1, random_state=42)
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nSVM Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# Random Forest
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nRandom Forest Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# Logistic Regression
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# XGBoost
classifier = XGBClassifier(random_state=42)
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nXGBoost Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# KNN
classifier = KNeighborsClassifier()
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nKNN Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# Decision Tree
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_undersampled, y_train_undersampled)
y_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nDecision Tree Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)

# Neural Network
classifier = MLPClassifier(hidden_layer_sizes=(64, 128), random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# Step 19: Evaluate the Neural Network classifier
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)

print("\nNeural Network Classification Results:")
print(f"Accuracy: {accuracy:.2f}%")
print(f"F1 Score: {f1:.2f}")
print(f"AUC Score: {auc_score:.2f}")
print("Confusion Matrix:")
print(confusion_mat)
