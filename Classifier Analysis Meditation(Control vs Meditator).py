#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('AnalysisMed.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()
data['Group'] = label_encoder.fit_transform(data['Group'])

# Define your feature columns (X) and target columns (y_group)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]
y_group = data['Group']

# Split the data into training and testing sets for Group classification
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X, y_group, test_size=0.1, random_state=23)

# Standardize the features for Group classification
scaler_group = StandardScaler()
X_train_group = scaler_group.fit_transform(X_train_group)
X_test_group = scaler_group.transform(X_test_group)

# Create and fit a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_group, y_train_group)

# Create and fit a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_group, y_train_group)

# Create and fit an SVM model
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_group, y_train_group)

# Create and fit a Deep Learning (Neural Network) model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train_group.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X_train_group, y_train_group, epochs=50, batch_size=32)

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(X_test_group)
accuracy_rf = accuracy_score(y_test_group, y_pred_rf)
report_rf = classification_report(y_test_group, y_pred_rf)

# Predict using the Decision Tree model
y_pred_dt = dt_model.predict(X_test_group)
accuracy_dt = accuracy_score(y_test_group, y_pred_dt)
report_dt = classification_report(y_test_group, y_pred_dt)

# Predict using the SVM model
y_pred_svm = svm_model.predict(X_test_group)
accuracy_svm = accuracy_score(y_test_group, y_pred_svm)
report_svm = classification_report(y_test_group, y_pred_svm)

# Predict using the Deep Learning (Neural Network) model
y_pred_nn = model.predict(X_test_group)
accuracy_nn = accuracy_score(y_test_group, (y_pred_nn > 0.5))
report_nn = classification_report(y_test_group, (y_pred_nn > 0.5))

# Print the results
print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_rf}')
print(report_rf)

print("Decision Tree Classifier:")
print(f'Accuracy: {accuracy_dt}')
print(report_dt)

print("SVM Classifier:")
print(f'Accuracy: {accuracy_svm}')
print(report_svm)

print("Neural Network (Deep Learning) Classifier:")
print(f'Accuracy: {accuracy_nn}')
print(report_nn)


# In[2]:


# Import the necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the dataset
data = pd.read_csv('AnalysisMed.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()
data['Group'] = label_encoder.fit_transform(data['Group'])

# Define your feature columns (X) and target columns (y_group)
X = data[["Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7","Slope_Channel_8","Slope_Channel_9","Slope_Channel_10","Slope_Channel_11","Slope_Channel_12","Slope_Channel_13","ITF_Channel_0","ITF_Channel_1","ITF_Channel_2","ITF_Channel_3","ITF_Channel_4","ITF_Channel_5","ITF_Channel_6","ITF_Channel_7","ITF_Channel_8","ITF_Channel_9","ITF_Channel_10","ITF_Channel_11","ITF_Channel_12","ITF_Channel_13","ITP_Channel_0","ITP_Channel_1","ITP_Channel_2","ITP_Channel_3","ITP_Channel_4","ITP_Channel_5","ITP_Channel_6","ITP_Channel_7","ITP_Channel_8","ITP_Channel_9","ITP_Channel_10","ITP_Channel_11","ITP_Channel_12","ITP_Channel_13","IAF_Channel_0","IAF_Channel_1","IAF_Channel_2","IAF_Channel_3","IAF_Channel_4","IAF_Channel_5","IAF_Channel_6","IAF_Channel_7","IAF_Channel_8","IAF_Channel_9","IAF_Channel_10","IAF_Channel_11","IAF_Channel_12","IAF_Channel_13","IAP_Channel_0","IAP_Channel_1","IAP_Channel_2","IAP_Channel_3","IAP_Channel_4","IAP_Channel_5","IAP_Channel_6","IAP_Channel_7","IAP_Channel_8","IAP_Channel_9","IAP_Channel_10","IAP_Channel_11","IAP_Channel_12","IAP_Channel_13","IBF_Channel_0","IBF_Channel_1","IBF_Channel_2","IBF_Channel_3","IBF_Channel_4","IBF_Channel_5","IBF_Channel_6","IBF_Channel_7","IBF_Channel_8","IBF_Channel_9","IBF_Channel_10","IBF_Channel_11","IBF_Channel_12","IBF_Channel_13","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IBP_Channel_6","IBP_Channel_7","IBP_Channel_8","IBP_Channel_9","IBP_Channel_10","IBP_Channel_11","IBP_Channel_12","IBP_Channel_13","Mean_PSD_Channel_Alpha_0","Mean_PSD_Channel_Alpha_1","Mean_PSD_Channel_Alpha_2","Mean_PSD_Channel_Alpha_3","Mean_PSD_Channel_Alpha_4","Mean_PSD_Channel_Alpha_5","Mean_PSD_Channel_Alpha_6","Mean_PSD_Channel_Alpha_7","Mean_PSD_Channel_Alpha_8","Mean_PSD_Channel_Alpha_9","Mean_PSD_Channel_Alpha_10","Mean_PSD_Channel_Alpha_11","Mean_PSD_Channel_Alpha_12","Mean_PSD_Channel_Alpha_13","Mean_PSD_Channel_Beta_0","Mean_PSD_Channel_Beta_1","Mean_PSD_Channel_Beta_2","Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7","Mean_PSD_Channel_Beta_8","Mean_PSD_Channel_Beta_9","Mean_PSD_Channel_Beta_10","Mean_PSD_Channel_Beta_11","Mean_PSD_Channel_Beta_12","Mean_PSD_Channel_Beta_13"]]
y_group = data['Group']

# Split the data into training and testing sets for Group classification
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X, y_group, test_size=0.1, random_state=23)

# Standardize the features for Group classification
scaler_group = StandardScaler()
X_train_group = scaler_group.fit_transform(X_train_group)
X_test_group = scaler_group.transform(X_test_group)

# Create and fit a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_group, y_train_group)

# Create and fit a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_group, y_train_group)

# Create and fit an SVM model
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_group, y_train_group)

# Create and fit a Deep Learning (Neural Network) model
model = Sequential()
model.add(Dense(units=64, input_dim=X_train_group.shape[1], activation='relu'))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
model.fit(X_train_group, y_train_group, epochs=50, batch_size=32)

# Predict using the Random Forest model
y_pred_rf = rf_model.predict(X_test_group)
accuracy_rf = accuracy_score(y_test_group, y_pred_rf)
report_rf = classification_report(y_test_group, y_pred_rf)

# Predict using the Decision Tree model
y_pred_dt = dt_model.predict(X_test_group)
accuracy_dt = accuracy_score(y_test_group, y_pred_dt)
report_dt = classification_report(y_test_group, y_pred_dt)

# Predict using the SVM model
y_pred_svm = svm_model.predict(X_test_group)
accuracy_svm = accuracy_score(y_test_group, y_pred_svm)
report_svm = classification_report(y_test_group, y_pred_svm)

# Predict using the Deep Learning (Neural Network) model
y_pred_nn = model.predict(X_test_group)
accuracy_nn = accuracy_score(y_test_group, (y_pred_nn > 0.5))
report_nn = classification_report(y_test_group, (y_pred_nn > 0.5))

# Print the results
print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_rf}')
print(report_rf)

print("Decision Tree Classifier:")
print(f'Accuracy: {accuracy_dt}')
print(report_dt)

print("SVM Classifier:")
print(f'Accuracy: {accuracy_svm}')
print(report_svm)

print("Neural Network (Deep Learning) Classifier:")
print(f'Accuracy: {accuracy_nn}')
print(report_nn)


# In[3]:


from sklearn.model_selection import cross_val_score, StratifiedKFold

# Create the Decision Tree model
model = DecisionTreeClassifier()

# Define the number of folds and a stratified splitter
num_folds = 5
cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

# Perform cross-validation and calculate accuracy
scores = cross_val_score(model, X, y_group, cv=cv, scoring='accuracy')

# Print the cross-validation results
print("Cross-Validation Accuracy (k-fold):", scores)
print("Mean Accuracy:", scores.mean())
print("Standard Deviation:", scores.std())


# In[4]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets for Random Forest
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X, y_group, test_size=0.1, random_state=23)

# Create and fit a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)

# Evaluate on the training data
y_train_pred_rf = rf_model.predict(X_train_rf)
accuracy_train_rf = accuracy_score(y_train_rf, y_train_pred_rf)

# Evaluate on the testing data
y_test_pred_rf = rf_model.predict(X_test_rf)
accuracy_test_rf = accuracy_score(y_test_rf, y_test_pred_rf)
report_test_rf = classification_report(y_test_rf, y_test_pred_rf)

# Cross-validation for Random Forest
scores_rf = cross_val_score(rf_model, X, y_group, cv=5, scoring='accuracy')

# Print the results
print("Random Forest Overfitting Analysis:")
print(f"Training Accuracy: {accuracy_train_rf}")
print(f"Testing Accuracy: {accuracy_test_rf}")
print(report_test_rf)
print("Cross-Validation Accuracy (k-fold):", scores_rf)
print("Mean Accuracy:", scores_rf.mean())
print("Standard Deviation:", scores_rf.std())


# In[5]:


from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Split the data into training and testing sets for SVM
X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X, y_group, test_size=0.1, random_state=23)

# Create and fit an SVM model
svm_model = SVC(kernel='linear', C=0.5, random_state=30)
svm_model.fit(X_train_svm, y_train_svm)

# Evaluate on the training data
y_train_pred_svm = svm_model.predict(X_train_svm)
accuracy_train_svm = accuracy_score(y_train_svm, y_train_pred_svm)

# Evaluate on the testing data
y_test_pred_svm = svm_model.predict(X_test_svm)
accuracy_test_svm = accuracy_score(y_test_svm, y_test_pred_svm)
report_test_svm = classification_report(y_test_svm, y_test_pred_svm)

# Cross-validation for SVM
scores_svm = cross_val_score(svm_model, X, y_group, cv=5, scoring='accuracy')

# Print the results
print("SVM Overfitting Analysis:")
print(f"Training Accuracy: {accuracy_train_svm}")
print(f"Testing Accuracy: {accuracy_test_svm}")
print(report_test_svm)
print("Cross-Validation Accuracy (k-fold):", scores_svm)
print("Mean Accuracy:", scores_svm.mean())
print("Standard Deviation:", scores_svm.std())


# In[6]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Split the data into training and testing sets for Decision Tree
X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X, y_group, test_size=0.1, random_state=23)

# Create and fit a Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_dt, y_train_dt)

# Evaluate on the training data
y_train_pred_dt = dt_model.predict(X_train_dt)
accuracy_train_dt = accuracy_score(y_train_dt, y_train_pred_dt)

# Evaluate on the testing data
y_test_pred_dt = dt_model.predict(X_test_dt)
accuracy_test_dt = accuracy_score(y_test_dt, y_test_pred_dt)
report_test_dt = classification_report(y_test_dt, y_test_pred_dt)

# Cross-validation for Decision Tree
scores_dt = cross_val_score(dt_model, X, y_group, cv=5, scoring='accuracy')

# Print the results
print("Decision Tree Overfitting Analysis:")
print(f"Training Accuracy: {accuracy_train_dt}")
print(f"Testing Accuracy: {accuracy_test_dt}")
print(report_test_dt)
print("Cross-Validation Accuracy (k-fold):", scores_dt)
print("Mean Accuracy:", scores_dt.mean())
print("Standard Deviation:", scores_dt.std())


# In[ ]:




