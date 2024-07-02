#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('Control Med&Rest.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()

# Encode the 'Task' column
data['Task'] = label_encoder.fit_transform(data['Task'])



# Define your feature columns (X) and target columns (y_group and y_task)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]

y_task = data['Task']

# Split the data into training and testing sets for Task classification
X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(X, y_task, test_size=0.1, random_state=21)

# Standardize the features for Task classification
scaler_task = StandardScaler()
X_train_task = scaler_task.fit_transform(X_train_task)
X_test_task = scaler_task.transform(X_test_task)

# Apply PCA for dimensionality reduction for Task classification
n_components = 21
pca = PCA(n_components=n_components)
X_train_task_pca = pca.fit_transform(X_train_task)
X_test_task_pca = pca.transform(X_test_task)

# Feature selection using SelectKBest with mutual information score on PCA components
k_best = SelectKBest(mutual_info_classif, k=21)
X_train_task_pca_best = k_best.fit_transform(X_train_task_pca, y_train_task)
X_test_task_pca_best = k_best.transform(X_test_task_pca)

# Define the SVM model for Task classification with PCA and selected features
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data for Task classification
y_pred_task_pca_best = svm_model.predict(X_test_task_pca_best)

# Calculate accuracy and classification report for Task classification
accuracy_task_pca_best = accuracy_score(y_test_task, y_pred_task_pca_best)
report_task_pca_best = classification_report(y_test_task, y_pred_task_pca_best)

print(f'Accuracy for Task Classification with PCA and Selected Features using SVM: {accuracy_task_pca_best}')
print(report_task_pca_best)


# In[2]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
confusion_matrix_task = confusion_matrix(y_test_task, y_pred_task_pca_best)

# Display the coanfusion matriacx
print("Confusion Matrix:")
print(confusion_matrix_task)


# In[3]:


from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data
y_pred_rf = rf_classifier.predict(X_test_task_pca_best)

# Evaluate the Random Forest classifier
accuracy_rf = accuracy_score(y_test_task, y_pred_rf)
report_rf = classification_report(y_test_task, y_pred_rf)

print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_rf}')
print(report_rf)


# In[4]:


from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
dt_classifier.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data
y_pred_dt = dt_classifier.predict(X_test_task_pca_best)

# Evaluate the Decision Tree classifier
accuracy_dt = accuracy_score(y_test_task, y_pred_dt)
report_dt = classification_report(y_test_task, y_pred_dt)

print("Decision Tree Classifier:")
print(f'Accuracy: {accuracy_dt}')
print(report_dt)


# In[5]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# Create a simple neural network model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train_task_pca_best.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_task_pca_best, y_train_task, epochs=25, batch_size=32)

# Predict the labels on the test data
y_pred_nn = model.predict(X_test_task_pca_best)
y_pred_nn = (y_pred_nn > 0.5).astype(int)  # Convert probabilities to class labels

# Print the classification report
report_nn = classification_report(y_test_task, y_pred_nn)
print("Neural Network (Deep Learning) Classifier:")
print(report_nn)

# Evaluate the model on the test data
_, accuracy_nn = model.evaluate(X_test_task_pca_best, y_test_task)
print(f'Accuracy: {accuracy_nn}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[6]:


# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('Med Med&Rest0311.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()

# Encode the 'Task' column
data['Task'] = label_encoder.fit_transform(data['Task'])



# Define your feature columns (X) and target columns (y_group and y_task)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]

y_task = data['Task']

# Split the data into training and testing sets for Task classification
X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(X, y_task, test_size=0.1, random_state=21)

# Standardize the features for Task classification
scaler_task = StandardScaler()
X_train_task = scaler_task.fit_transform(X_train_task)
X_test_task = scaler_task.transform(X_test_task)

# Apply PCA for dimensionality reduction for Task classification
n_components = 21
pca = PCA(n_components=n_components)
X_train_task_pca = pca.fit_transform(X_train_task)
X_test_task_pca = pca.transform(X_test_task)

# Feature selection using SelectKBest with mutual information score on PCA components
k_best = SelectKBest(mutual_info_classif, k=18)
X_train_task_pca_best = k_best.fit_transform(X_train_task_pca, y_train_task)
X_test_task_pca_best = k_best.transform(X_test_task_pca)

# Define the SVM model for Task classification with PCA and selected features
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data for Task classification
y_pred_task_pca_best = svm_model.predict(X_test_task_pca_best)

# Calculate accuracy and classification report for Task classification
accuracy_task_pca_best = accuracy_score(y_test_task, y_pred_task_pca_best)
report_task_pca_best = classification_report(y_test_task, y_pred_task_pca_best)

print(f'Accuracy for Task Classification with PCA and Selected Features using SVM: {accuracy_task_pca_best}')
print(report_task_pca_best)


# In[7]:


from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
confusion_matrix_task = confusion_matrix(y_test_task, y_pred_task_pca_best)

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix_task)


# In[8]:


from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data
y_pred_rf = rf_classifier.predict(X_test_task_pca_best)

# Evaluate the Random Forest classifier
accuracy_rf = accuracy_score(y_test_task, y_pred_rf)
report_rf = classification_report(y_test_task, y_pred_rf)

print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_rf}')
print(report_rf)


# In[9]:


from sklearn.tree import DecisionTreeClassifier

# Create a Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
dt_classifier.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data
y_pred_dt = dt_classifier.predict(X_test_task_pca_best)

# Evaluate the Decision Tree classifier
accuracy_dt = accuracy_score(y_test_task, y_pred_dt)
report_dt = classification_report(y_test_task, y_pred_dt)

print("Decision Tree Classifier:")
print(f'Accuracy: {accuracy_dt}')
print(report_dt)


# In[10]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report

# Create a simple neural network model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train_task_pca_best.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_task_pca_best, y_train_task, epochs=25, batch_size=32)

# Predict the labels on the test data
y_pred_nn = model.predict(X_test_task_pca_best)
y_pred_nn = (y_pred_nn > 0.5).astype(int)  # Convert probabilities to class labels

# Print the classification report
report_nn = classification_report(y_test_task, y_pred_nn)
print("Neural Network (Deep Learning) Classifier:")
print(report_nn)

# Evaluate the model on the test data
_, accuracy_nn = model.evaluate(X_test_task_pca_best, y_test_task)
print(f'Accuracy: {accuracy_nn}')


# In[11]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Create a simple neural network model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train_task_pca_best.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Fit the model to the training data
model.fit(X_train_task_pca_best, y_train_task, epochs=25, batch_size=32)

# Evaluate the model on the test data
_, accuracy_nn = model.evaluate(X_test_task_pca_best, y_test_task)
print("Neural Network (Deep Learning) Classifier:")
print(f'Accuracy: {accuracy_nn}')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[12]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC


# Load the dataset
data = pd.read_csv('Control Med&Rest.csv')


# Encode the 'Task' column
data['Task'] = label_encoder.fit_transform(data['Task'])

# Define your feature columns (X) and target columns (y_group and y_task)
# Define your feature columns (X) and target columns (y_group and y_task)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]

y_task = data['Task']

# Split the data into training and testing sets for Task classification
X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(X, y_task, test_size=0.1, random_state=23)

# Standardize the features for Task classification
scaler_task = StandardScaler()
X_train_task = scaler_task.fit_transform(X_train_task)
X_test_task = scaler_task.transform(X_test_task)

# Apply PCA for dimensionality reduction for Task classification
n_components = 21
pca = PCA(n_components=n_components)
X_train_task_pca = pca.fit_transform(X_train_task)
X_test_task_pca = pca.transform(X_test_task)

# Feature selection using SelectKBest with mutual information score on PCA components
k_best = SelectKBest(mutual_info_classif, k=14)
X_train_task_pca_best = k_best.fit_transform(X_train_task_pca, y_train_task)
X_test_task_pca_best = k_best.transform(X_test_task_pca)

# Define the SVM model for Task classification with PCA and selected features
svm_model = SVC(kernel='linear', C=0.5, random_state=30)

# Implement k-fold cross-validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=40)  # You can adjust the number of splits as needed

# Perform cross-validation and calculate accuracy
cross_val_scores = cross_val_score(svm_model, X_train_task_pca_best, y_train_task, cv=k_fold, scoring='accuracy')

# Print the cross-validation accuracy scores
print("Cross-Validation Accuracy Scores:", cross_val_scores)

# Find the best accuracy and the corresponding model
best_accuracy = max(cross_val_scores)
best_model_index = cross_val_scores.tolist().index(best_accuracy)
best_model = SVC(kernel='linear', C=0.5, random_state=30)  # Create a new SVM model with the best parameters
best_model.fit(X_train_task_pca_best, y_train_task)

# Evaluate the best model on the test set
y_pred_best_model = best_model.predict(X_test_task_pca_best)
accuracy_best_model = accuracy_score(y_test_task, y_pred_best_model)
report_best_model = classification_report(y_test_task, y_pred_best_model)

print("Best Cross-Validation Accuracy:", best_accuracy)
print(report_best_model)


# In[ ]:





# In[13]:


# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

# Load the dataset
data = pd.read_csv('Control Med&Rest.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()

# Encode the 'Task' column
data['Task'] = label_encoder.fit_transform(data['Task'])



# Define your feature columns (X) and target columns (y_group and y_task)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]

y_task = data['Task']

# Split the data into training and testing sets for Task classification
X_train_task, X_test_task, y_train_task, y_test_task = train_test_split(X, y_task, test_size=0.1, random_state=21)

# Standardize the features for Task classification
scaler_task = StandardScaler()
X_train_task = scaler_task.fit_transform(X_train_task)
X_test_task = scaler_task.transform(X_test_task)

# Apply PCA for dimensionality reduction for Task classification
n_components = 24
pca = PCA(n_components=n_components)
X_train_task_pca = pca.fit_transform(X_train_task)
X_test_task_pca = pca.transform(X_test_task)

# Feature selection using SelectKBest with mutual information score on PCA components
k_best = SelectKBest(mutual_info_classif, k=21)
X_train_task_pca_best = k_best.fit_transform(X_train_task_pca, y_train_task)
X_test_task_pca_best = k_best.transform(X_test_task_pca)

# Define the SVM model for Task classification with PCA and selected features
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_task_pca_best, y_train_task)

# Predict the labels on the test data for Task classification
y_pred_task_pca_best = svm_model.predict(X_test_task_pca_best)

# Calculate accuracy and classification report for Task classification
accuracy_task_pca_best = accuracy_score(y_test_task, y_pred_task_pca_best)
report_task_pca_best = classification_report(y_test_task, y_pred_task_pca_best)

print(f'Accuracy for Task Classification with PCA and Selected Features using SVM: {accuracy_task_pca_best}')
print(report_task_pca_best)

