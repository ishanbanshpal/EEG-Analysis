#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # Import SelectKBest and mutual_info_classif
from sklearn.decomposition import PCA  # Import PCA
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
data = pd.read_csv('Classifier8.csv')

# Encode the 'Group' column
label_encoder = LabelEncoder()
data['Group'] = label_encoder.fit_transform(data['Group'])

# Encode the 'Task' column
data['Task'] = label_encoder.fit_transform(data['Task'])

# Define your feature columns (X) and target columns (y_group and y_task)
X = data[["IBP_Channel_7","IBP_Channel_6","IBP_Channel_0","IBP_Channel_1","IBP_Channel_2","IBP_Channel_3","IBP_Channel_4","IBP_Channel_5","IAP_Channel_0", "IAP_Channel_1", "IAP_Channel_2", "IAP_Channel_3", "IAP_Channel_4", "IAP_Channel_5","IAP_Channel_6", "IAP_Channel_7","IBF_Channel_0", "IBF_Channel_1", "IBF_Channel_2", "IBF_Channel_3", "IBF_Channel_4", "IBF_Channel_5","IBF_Channel_6", "IBF_Channel_7","IAF_Channel_0", "IAF_Channel_1", "IAF_Channel_2", "IAF_Channel_3", "IAF_Channel_4", "IAF_Channel_5","IAF_Channel_6", "IAF_Channel_7","Slope_Channel_0","Slope_Channel_1","Slope_Channel_2","Slope_Channel_3","Slope_Channel_4","Slope_Channel_5","Slope_Channel_6","Slope_Channel_7",'Mean_PSD_Channel_Beta_1','Mean_PSD_Channel_Beta_2',"Mean_PSD_Channel_Beta_3","Mean_PSD_Channel_Beta_4","Mean_PSD_Channel_Beta_5","Mean_PSD_Channel_Beta_6","Mean_PSD_Channel_Beta_7",'Mean_PSD_Channel_Alpha_5','Mean_PSD_Channel_Alpha_4','Mean_PSD_Channel_Alpha_3','Mean_PSD_Channel_Alpha_2','Mean_PSD_Channel_Alpha_1','Mean_PSD_Channel_Alpha_6','Mean_PSD_Channel_Alpha_7']]

y_group = data['Group']
y_task = data['Task']

# Split the data into training and testing sets for Group classification
X_train_group, X_test_group, y_train_group, y_test_group = train_test_split(X, y_group, test_size=0.2, random_state=42)

# Standardize the features for Group classification
scaler_group = StandardScaler()
X_train_group = scaler_group.fit_transform(X_train_group)
X_test_group = scaler_group.transform(X_test_group)

# Apply PCA for dimensionality reduction
n_components = 16  # You can adjust the number of components
pca = PCA(n_components=n_components)

X_train_group_pca = pca.fit_transform(X_train_group)
X_test_group_pca = pca.transform(X_test_group)

# Feature selection using SelectKBest with mutual information score on PCA components
k_best = SelectKBest(mutual_info_classif, k=10)  # You can adjust the number of top features (k)
X_train_group_pca_best = k_best.fit_transform(X_train_group_pca, y_train_group)
X_test_group_pca_best = k_best.transform(X_test_group_pca)

# Define the deep learning model for Group classification with PCA and selected features
model_group_pca_best = keras.Sequential([
    layers.Input(shape=(X_train_group_pca_best.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(2, activation='softmax')  # Assuming 2 classes in Group
])

# Compile and train the model for Group classification with PCA and selected features
model_group_pca_best.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_group_pca_best.fit(X_train_group_pca_best, y_train_group, epochs=25, batch_size=25, validation_split=0.1, verbose=2)

# Evaluate the model for Group classification on the test set with PCA and selected features
y_pred_group_pca_best = np.argmax(model_group_pca_best.predict(X_test_group_pca_best), axis=-1)
accuracy_group_pca_best = accuracy_score(y_test_group, y_pred_group_pca_best)
report_group_pca_best = classification_report(y_test_group, y_pred_group_pca_best)

print(f'Accuracy for Group Classification with PCA and Selected Features: {accuracy_group_pca_best}')
print(report_group_pca_best)


# In[ ]:




