#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy.stats import mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests
import numpy as np
import math

# Load your dataset
data = pd.read_csv('Frameworkfeatures2.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_novice_rest': ((0, 1), (0, 0)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}


# Define percentages for subset creation
percentages = [0.20, 0.40, 0.60, 0.80, 1.0]  # Example percentages

# Function to calculate Hedges' g
def hedges_g(x, y):
    n1, n2 = len(x), len(y)
    dof = n1 + n2 - 2
    pooled_std = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / dof)
    g = (np.mean(x) - np.mean(y)) / pooled_std
    correction_factor = 1 - (3 / (4 * dof - 1))
    return g * correction_factor

# Perform statistical tests and report results for each condition and each subset percentage
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for percentage in percentages:
        print(f"Subset Percentage: {percentage * 100}%")
        
        # Determine the number of observations for each group based on the percentage
        group1_size = int(len(group1) * percentage)
        group2_size = int(len(group2) * percentage)
        
        # Randomly sample observations from each group
        group1_subset = group1.sample(n=group1_size, random_state=42)
        group2_subset = group2.sample(n=group2_size, random_state=42)
        
        # Initialize lists for storing effect sizes and their CIs
        feature_info = []
        
        # Iterate over features
        for feature in scaled_data.columns:
            if feature not in ['Level', 'EEG_State']:
                group1_values = group1_subset[feature].values
                group2_values = group2_subset[feature].values
                
                # Calculate effect size (Hedges' g)
                g = hedges_g(group1_values, group2_values)
                
                # Calculate standard error for Hedges' g
                n1, n2 = len(group1_values), len(group2_values)
                se = math.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
                
                # Calculate 95% confidence interval for effect size
                ci_lower = g - 1.96 * se
                ci_upper = g + 1.96 * se
                ci_width = ci_upper - ci_lower
                
                feature_info.append((feature, g, ci_lower, ci_upper, ci_width))
        
        # Sort features by the width of their confidence intervals
        feature_info.sort(key=lambda x: x[4])
        
        # Perform statistical tests on sorted features
        p_values = []
        significant_features = []
        for feature, g, ci_lower, ci_upper, ci_width in feature_info:
            group1_values = group1_subset[feature].values
            group2_values = group2_subset[feature].values

            # Perform Mann-Whitney U test for the distributions of the feature
            u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
            p_values.append(p_value)
        
        # Adjust p-value for multiple comparisons
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Determine if the feature is statistically significant based on corrected p-values
        for (feature, g, ci_lower, ci_upper, ci_width), corrected_p_value in zip(feature_info, corrected_p_values):
            if corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print(f"Effect sizes: {[(feature, g) for feature, g, _, _, _ in feature_info if feature in significant_features]}")
        print(f"Confidence Intervals (95%): {[(feature, (ci_lower, ci_upper)) for feature, _, ci_lower, ci_upper, _ in feature_info if feature in significant_features]}")
        print("\n")


# In[4]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy.stats import norm

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'novice_rest_vs_novice_meditation': ((0, 0), (0, 1)),
}

# Define percentages for subset creation
percentages = [0.20, 0.40, 0.60, 0.80, 1.0]  # Example percentages

# Perform statistical tests and report results for each condition and each subset percentage
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for percentage in percentages:
        print(f"Subset Percentage: {percentage * 100}%")
        
        # Determine the number of observations for each group based on the percentage
        group1_size = int(len(group1) * percentage)
        group2_size = int(len(group2) * percentage)
        
        # Randomly sample observations from each group
        group1_subset = group1.sample(n=group1_size, random_state=42)
        group2_subset = group2.sample(n=group2_size, random_state=42)
        
        # Initialize significant features counter
        significant_features = []
        
        # Initialize dictionaries to store effect sizes and confidence intervals
        effect_sizes = {}
        ci_lower = {}
        ci_upper = {}
        
        # Iterate over features
        p_values = []
        for feature in scaled_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                group1_values = group1_subset[feature].values
                group2_values = group2_subset[feature].values

                # Perform Mann-Whitney U test for the distributions of the feature
                u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                p_values.append(p_value)
                
                # Calculate effect size (Hedges' g)
                n1 = len(group1_values)
                n2 = len(group2_values)
                dof = n1 + n2 - 2
                u1 = (n1 * n2) / 2
                hedges_g = (u_stat / (n1 * n2)) * (2 / np.sqrt((dof + 1) * (1 / n1 + 1 / n2)))
                effect_sizes[feature] = hedges_g
                
                # Calculate confidence intervals for effect size
                se = np.sqrt((dof * (n1 + n2)) / (n1 * n2 * (n1 + n2 - 2)))
                ci = norm.ppf(0.975) * se
                ci_lower[feature] = hedges_g - ci
                ci_upper[feature] = hedges_g + ci

        # Adjust p-value for multiple comparisons
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Determine if the feature is statistically significant based on corrected p-values
        for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
            if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print("\n")
        print("Effect Sizes:")
        print(effect_sizes)
        print("\n")
        print("Confidence Intervals (95%):")
        print("Lower Bound:")
        print(ci_lower)
        print("\n")
        print("Upper Bound:")
        print(ci_upper)
        print("\n")


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'novice_rest_vs_novice_meditation': ((0, 0), (0, 1)),
}

# Define percentages for subset creation
percentages = [0.20, 0.40, 0.60, 0.80, 1.0]  # Example percentages

# Perform statistical tests and report results for each condition and each subset percentage
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for percentage in percentages:
        print(f"Subset Percentage: {percentage * 100}%")
        
        # Determine the number of observations for each group based on the percentage
        group1_size = int(len(group1) * percentage)
        group2_size = int(len(group2) * percentage)
        
        # Randomly sample observations from each group
        group1_subset = group1.sample(n=group1_size, random_state=42)
        group2_subset = group2.sample(n=group2_size, random_state=42)
        
        # Initialize significant features counter
        significant_features = []
        
        # Iterate over features
        p_values = []
        for feature in scaled_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                group1_values = group1_subset[feature].values
                group2_values = group2_subset[feature].values

                # Perform Mann-Whitney U test for the distributions of the feature
                u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                p_values.append(p_value)

        # Adjust p-value for multiple comparisons
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Determine if the feature is statistically significant based on corrected p-values
        for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
            if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print("\n")


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Define percentages for subset creation
percentages = [0.20, 0.40, 0.60, 0.80, 1.0]  # Example percentages

def stratified_sample(df, stratify_cols, frac, random_state=None):
    return df.groupby(stratify_cols, group_keys=False).apply(lambda x: x.sample(frac=frac, random_state=random_state))

# Perform statistical tests and report results for each condition and each subset percentage
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for percentage in percentages:
        print(f"Subset Percentage: {percentage * 100}%")
        
        # Calculate the fraction for stratified sampling
        frac = percentage
        
        # Perform stratified sampling
        group1_subset = stratified_sample(group1, ['Level', 'EEG_State'], frac, random_state=42)
        group2_subset = stratified_sample(group2, ['Level', 'EEG_State'], frac, random_state=42)
        
        # Initialize significant features counter
        significant_features = []
        
        # Iterate over features
        p_values = []
        for feature in scaled_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                group1_values = group1_subset[feature].values
                group2_values = group2_subset[feature].values

                # Perform Mann-Whitney U test for the distributions of the feature
                u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                p_values.append(p_value)

        # Adjust p-value for multiple comparisons using Benjamini-Hochberg correction
        _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

        # Determine if the feature is statistically significant based on corrected p-values
        for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
            if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print("\n")


# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import numpy as np

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Define percentages for subset creation
percentages = [0.20, 0.40, 0.60, 0.80, 1.0]  # Example percentages

# Perform statistical tests and report results for each condition and each subset percentage
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for percentage in percentages:
        print(f"Subset Percentage: {percentage * 100}%")
        
        # Determine the number of observations for each group based on the percentage
        group1_size = int(len(group1) * percentage)
        group2_size = int(len(group2) * percentage)
        
        # Randomly sample observations from each group
        group1_subset = group1.sample(n=group1_size, random_state=42)
        group2_subset = group2.sample(n=group2_size, random_state=42)
        
        # Initialize significant features counter
        significant_features = []
        
        # Iterate over features
        p_values = []
        for feature in scaled_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                group1_values = group1_subset[feature].values
                group2_values = group2_subset[feature].values

                # Perform Mann-Whitney U test for the distributions of the feature
                u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                p_values.append(p_value)

        # Adjust p-value for multiple comparisons
        _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

        # Determine if the feature is statistically significant based on corrected p-values
        for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
            if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print("\n")


# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Encode categorical variables
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the data
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])


# In[2]:


from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

significant_features = []

for condition, (group1_condition, group2_condition) in conditions.items():
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]

    p_values = []
    for feature in scaled_data.columns:
        if feature not in ['Level', 'EEG_State']:
            group1_values = group1[feature].values
            group2_values = group2[feature].values
            _, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
            p_values.append(p_value)

    _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

    for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
        if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
            significant_features.append(feature)


# In[3]:


# Define the target variable and features
X = scaled_data[significant_features]
y = data['Level']  # or another target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[4]:


# Define the target variable and features
X = scaled_data[significant_features]
y = data['EEG_State']  # or another target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")


# In[5]:


from sklearn.multioutput import MultiOutputClassifier

# Target variables
y = data[['Level', 'EEG_State']]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Multi-output decision tree
clf_multi = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
clf_multi.fit(X_train, y_train)

# Predict and evaluate
y_pred_multi = clf_multi.predict(X_test)

accuracy_level_multi = accuracy_score(y_test['Level'], y_pred_multi[:, 0])
accuracy_eeg_multi = accuracy_score(y_test['EEG_State'], y_pred_multi[:, 1])
print(f"Accuracy for predicting Level (multi-output): {accuracy_level_multi}")
print(f"Accuracy for predicting EEG_State (multi-output): {accuracy_eeg_multi}")


# In[ ]:





# In[ ]:





# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform statistical tests and report results for each condition
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(scaled_data['Level'] == group1_condition[0]) & (scaled_data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(scaled_data['Level'] == group2_condition[0]) & (scaled_data['EEG_State'] == group2_condition[1])]
    
    # Iterate over features
    significant_features = []
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1_feature = group1[feature]
            group2_feature = group2[feature]
            
            # Check if both groups have a non-zero size
            if len(group1_feature) > 0 and len(group2_feature) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(group1_feature, group2_feature)
                
                # Adjust p-value for multiple comparisons
                _, corrected_p_value, _, _ = multipletests([p_value], method='bonferroni')
                
                # Check significance level (e.g., alpha = 0.05)
                if corrected_p_value < 0.05:
                    significant_features.append(feature)
            else:
                print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")
    
    # Report results
    print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")
    print("\n")


# In[7]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform statistical tests and report results for each condition
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    # Initialize significant features counter
    significant_features = []
    
    # Iterate over features
    p_values = []
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1_values = group1[feature].values
            group2_values = group2[feature].values

            # Perform Mann-Whitney U test for the distributions of the feature
            u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
            p_values.append(p_value)

    # Adjust p-value for multiple comparisons
    _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

    # Determine if the feature is statistically significant based on corrected p-values
    for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
        if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
            significant_features.append(feature)

    # Report results
    print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")
    print("\n")


# In[8]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform statistical tests and report results for each condition
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    # Initialize significant features counter
    significant_features = []
    
    # Iterate over features
    p_values = []
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1_values = group1[feature].values
            group2_values = group2[feature].values

            # Perform Mann-Whitney U test for the distributions of the feature
            u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
            p_values.append(p_value)

    # Adjust p-value for multiple comparisons
    _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

    # Determine if the feature is statistically significant based on corrected p-values
    for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
        if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
            significant_features.append(feature)

    # Report results
    print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")
    print("\n")


# In[ ]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
import itertools

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data.drop(columns=["Filename"])), columns=data.columns[1:])

# Define conditions for grouping
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Define range of subset sizes
subset_sizes = [15, 20, 25, 29]  # Example subset sizes

# Perform statistical tests and report results for each condition and each subset size
for condition, (group1_condition, group2_condition) in conditions.items():
    print(f"Condition: {condition}")
    
    # Filter data based on conditions
    group1 = scaled_data[(data['Level'] == group1_condition[0]) & (data['EEG_State'] == group1_condition[1])]
    group2 = scaled_data[(data['Level'] == group2_condition[0]) & (data['EEG_State'] == group2_condition[1])]
    
    for subset_size in subset_sizes:
        print(f"Subset Size: {subset_size}")
        
        # Generate all combinations of subset indices for each group
        group1_subsets = list(itertools.combinations(group1.index, subset_size))
        group2_subsets = list(itertools.combinations(group2.index, subset_size))
        
        # Initialize significant features counter
        significant_features = []
        
        # Iterate over features
        p_values = []
        for feature in scaled_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                for subset1, subset2 in itertools.product(group1_subsets, group2_subsets):
                    group1_values = scaled_data.loc[subset1, feature].values
                    group2_values = scaled_data.loc[subset2, feature].values

                    # Perform Mann-Whitney U test for the distributions of the feature
                    u_stat, p_value = mannwhitneyu(group1_values, group2_values, alternative='two-sided')
                    p_values.append(p_value)

        # Adjust p-value for multiple comparisons
        _, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

        # Determine if the feature is statistically significant based on corrected p-values
        for feature, corrected_p_value in zip(scaled_data.columns, corrected_p_values):
            if feature not in ['Level', 'EEG_State'] and corrected_p_value < 0.05:
                significant_features.append(feature)

        # Report results
        print(f"Number of features included in analysis: {len(scaled_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
        print(f"Number of statistically significant features: {len(significant_features)}")
        print(f"Significant features: {significant_features}")
        print("\n")


# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Check the conditions used for grouping
print("Conditions used for grouping:")
for condition, ((level1, state1), (level2, state2)) in conditions.items():
    print(f"{condition}: Group 1 - Level: {level1}, EEG State: {state1} | Group 2 - Level: {level2}, EEG State: {state2}")

# Examine the dataset for patterns or imbalances
print("\nDataset information:")
print(data.info())
print("\nDistribution of samples across groups:")
print(data.groupby(['Level', 'EEG_State']).size())

# Adjust conditions or data preprocessing if needed
# You may need to revise the conditions or preprocess the data to address any imbalances
# For example, you could adjust the conditions or oversample/undersample certain groups

# Preprocessing
# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define subsets based on conditions
subsets = scaled_data.groupby(['Level', 'EEG_State'])

# Perform statistical tests and report results for each subset
for subset_name, subset_data in subsets:
    print(f"\nSubset: {subset_name}")
    
    # Perform Mann-Whitney U test for each feature
    significant_features = []
    for feature in subset_data.columns:
        # Exclude 'Level', 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1 = subset_data[feature][(subset_data['Level'] == subset_name[0]) & (subset_data['EEG_State'] == subset_name[1])]
            group2 = subset_data[feature][(subset_data['Level'] != subset_name[0]) | (subset_data['EEG_State'] != subset_name[1])]
            
            # Check if both groups have a non-zero size
            if len(group1) > 0 and len(group2) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(group1, group2)
                
                # Adjust p-value for multiple comparisons
                _, corrected_p_value, _, _ = multipletests([p_value], method='bonferroni')
                
                # Check significance level (e.g., alpha = 0.05)
                if corrected_p_value < 0.05:
                    significant_features.append(feature)
            else:
                print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")
    
    # Report results
    print(f"Number of features included in analysis: {len(subset_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")


# In[5]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform feature selection for each condition and each filename
significant_features = {}

# Iterate over each condition
for condition, ((level1, state1), (level2, state2)) in conditions.items():
    significant_features[condition] = {}
    
    # Iterate over each unique filename
    for filename in filenames.unique():
        # Select data for the current filename
        subset_data = scaled_data[filenames == filename]
        
        # Split the subset data into groups based on conditions
        group1 = subset_data[(data['Level'] == level1) & (data['EEG_State'] == state1)]
        group2 = subset_data[(data['Level'] == level2) & (data['EEG_State'] == state2)]
        
        # Perform Mann-Whitney U test for each feature
        significant_features[condition][filename] = []
        for feature in subset_data.columns:
            # Exclude 'Level' and 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                # Check if both groups have non-zero size
                if len(group1) > 0 and len(group2) > 0:
                    # Perform Mann-Whitney U test
                    u_stat, p_value = mannwhitneyu(group1[feature], group2[feature])
                    
                    # Adjust p-value for multiple comparisons
                    _, corrected_p_value, _, _ = multipletests([p_value], method='bonferroni')
                    
                    # Check significance level (e.g., alpha = 0.05)
                    if corrected_p_value < 0.05:
                        significant_features[condition][filename].append(feature)
                else:
                    print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")

# Report significant features for each condition and filename
for condition, filename_data in significant_features.items():
    print(f"Condition: {condition}")
    for filename, features_data in filename_data.items():
        print(f"Filename: {filename}")
        print(f"Significant features: {features_data}")
        print(f"Number of statistically significant features: {len(features_data)}")
        print("\n")


# In[4]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define subsets based on observations
# Example: Group by 'Level' and 'EEG_State'
subsets = scaled_data.groupby(['Level', 'EEG_State'])

# Perform statistical tests and report results for each subset
for subset_name, subset_data in subsets:
    print(f"Subset: {subset_name}")
    
    # Perform Mann-Whitney U test for each feature
    significant_features = []
    for feature in subset_data.columns:
        # Exclude 'Level', 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1 = subset_data[feature][(subset_data['Level'] == subset_name[0]) & (subset_data['EEG_State'] == subset_name[1])]
            group2 = subset_data[feature][(subset_data['Level'] != subset_name[0]) | (subset_data['EEG_State'] != subset_name[1])]
            
            # Check size of each group
            print(f"Size of group1: {len(group1)}")
            print(f"Size of group2: {len(group2)}")
            
            # Check if both groups have a non-zero size
            if len(group1) > 0 and len(group2) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(group1, group2)
                
                # Adjust p-value for multiple comparisons
                _, corrected_p_value, _, _ = multipletests([p_value], method='bonferroni')
                
                # Check significance level (e.g., alpha = 0.05)
                if corrected_p_value < 0.05:
                    significant_features.append(feature)
            else:
                print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")
    
    # Report results
    print(f"Number of features included in analysis: {len(subset_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")
    print("\n")


# In[3]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define subsets based on observations
# Example: Group by 'Level' and 'EEG_State'
subsets = scaled_data.groupby(['Level', 'EEG_State'])

# Perform statistical tests and report results for each subset
for subset_name, subset_data in subsets:
    print(f"Subset: {subset_name}")
    
    # Perform Mann-Whitney U test for each feature
    significant_features = []
    for feature in subset_data.columns:
        # Exclude 'Level', 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            group1 = subset_data[feature][(subset_data['Level'] == subset_name[0]) & (subset_data['EEG_State'] == subset_name[1])]
            group2 = subset_data[feature][(subset_data['Level'] != subset_name[0]) & (subset_data['EEG_State'] != subset_name[1])]
            
            # Check if both groups have a non-zero size
            if len(group1) > 0 and len(group2) > 0:
                # Perform Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(group1, group2)
                
                # Adjust p-value for multiple comparisons
                _, corrected_p_value, _, _ = multipletests([p_value], method='bonferroni')
                
                # Check significance level (e.g., alpha = 0.05)
                if corrected_p_value < 0.05:
                    significant_features.append(feature)
            else:
                print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")
    
    # Report results
    print(f"Number of features included in analysis: {len(subset_data.columns) - 2}")  # Excluding 'Level' and 'EEG_State'
    print(f"Number of statistically significant features: {len(significant_features)}")
    print(f"Significant features: {significant_features}")
    print("\n")


# In[6]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define unique filenames
unique_filenames = filenames.unique()

print("Encoded values in EEG_State:", data['EEG_State'].unique())
print("Data type of EEG_State column:", data['EEG_State'].dtype)

# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform feature selection for each condition and each filename
significant_features = {}

# Iterate over each condition
for condition, ((level1, state1), (level2, state2)) in conditions.items():
    significant_features[condition] = {}
    
    # Iterate over each unique filename
    for filename in unique_filenames:
        # Select data for the current filename
        subset_data = scaled_data[filenames == filename]
        
        # Split the subset data into groups based on conditions
        group1 = subset_data[(data['Level'] == level1) & (data['EEG_State'] == state1)]
        group2 = subset_data[(data['Level'] == level2) & (data['EEG_State'] == state2)]
        
        # Debugging messages
        print(f"Condition: {condition}, Filename: {filename}")
        print(f"Size of group1: {len(group1)}")
        print(f"Size of group2: {len(group2)}")
        
        # Perform Mann-Whitney U test for each feature
        u_stats = []
        p_values = []
        for feature in data.columns:
            # Exclude 'Level', 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                # Check if group1 and group2 have a non-zero size
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_value = mannwhitneyu(group1[feature], group2[feature])
                    u_stats.append(u_stat)
                    p_values.append(p_value)
                else:
                    # Handle the case where one or both groups have zero size
                    print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")
        
        # Adjust p-values for multiple comparisons
        reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
        # Filter out 'Level' and 'EEG_State' from significant features
        significant_features[condition][filename] = {'significant_features': [scaled_data.columns[i] for i in range(len(reject)) if reject[i] and data.columns[i] not in ['Level', 'EEG_State', 'Filename']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition and filename
for condition, filename_data in significant_features.items():
    print(f"Condition: {condition}")
    for filename, features_data in filename_data.items():
        print(f"Filename: {filename}")
        print(f"Significant features: {features_data['significant_features']}")
        print(f"Number of statistically significant features: {len(features_data['significant_features'])}")


# In[8]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)]
novice_meditation = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]
experienced_rest = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]
experienced_meditation = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]

# Define the updated conditions
conditions = {
    'novice_rest_vs_experienced_rest': (novice_rest, experienced_rest),
    'novice_meditation_vs_experienced_meditation': (novice_meditation, experienced_meditation),
    'experienced_rest_vs_experienced_meditation': (experienced_rest, experienced_meditation),
    'experienced_meditation_vs_novice_meditation': (experienced_meditation, novice_meditation),
}

# Perform feature selection for each condition and each target variable
significant_features = {}

# Iterate over each condition
for condition, (group1, group2) in conditions.items():
    # Perform t-tests for EEG_State
    t_stats_eeg_state = []
    p_values_eeg_state = []
    for feature in scaled_data.columns:
        t_stat, p_value = ttest_ind(group1[feature], group2[feature])
        t_stats_eeg_state.append(t_stat)
        p_values_eeg_state.append(p_value)
    
    # Perform t-tests for Level
    t_stats_level = []
    p_values_level = []
    for feature in scaled_data.columns:
        t_stat, p_value = ttest_ind(group1[feature], group2[feature])
        t_stats_level.append(t_stat)
        p_values_level.append(p_value)
    
    # Adjust p-values for multiple comparisons for EEG_State
    reject_eeg_state, corrected_p_values_eeg_state, _, _ = multipletests(p_values_eeg_state, method='bonferroni')
    significant_features_eeg_state = [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject_eeg_state[i]]
    
    # Adjust p-values for multiple comparisons for Level
    reject_level, corrected_p_values_level, _, _ = multipletests(p_values_level, method='bonferroni')
    significant_features_level = [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject_level[i]]
    
    # Store the significant features for both EEG_State and Level
    significant_features[condition] = {'significant_features_eeg_state': significant_features_eeg_state, 
                                       'corrected_p_values_eeg_state': corrected_p_values_eeg_state,
                                       'significant_features_level': significant_features_level, 
                                       'corrected_p_values_level': corrected_p_values_level}

# Report significant features for each condition and each target variable
for condition in significant_features:
    print(f"Significant features for {condition} (EEG_State): {significant_features[condition]['significant_features_eeg_state']}")
    print(f"Significant features for {condition} (Level): {significant_features[condition]['significant_features_level']}")


# In[11]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)]
novice_meditation = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]
experienced_rest = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]
experienced_meditation = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]

# Define the updated conditions
conditions = {
    'novice_rest_vs_experienced_rest': (novice_rest, experienced_rest),
    'novice_meditation_vs_experienced_meditation': (novice_meditation, experienced_meditation),
    'experienced_rest_vs_experienced_meditation': (experienced_rest, experienced_meditation),
    'experienced_meditation_vs_novice_meditation': (experienced_meditation, novice_meditation),
}

# Perform feature selection for each condition
significant_features = {}

# Iterate over each condition
for condition, (group1, group2) in conditions.items():
    t_stats = []
    p_values = []
    # Perform t-tests for each feature
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            t_stat, p_value = ttest_ind(group1[feature], group2[feature])
            t_stats.append(t_stat)
            p_values.append(p_value)
    
    # Adjust p-values for multiple comparisons
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    significant_features[condition] = {'significant_features': [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject[i] and scaled_data.columns[i] not in ['Level', 'EEG_State']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition
for condition in significant_features:
    print(f"Significant features for {condition}: {significant_features[condition]['significant_features']}")


# In[10]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)]
novice_meditation = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]
experienced_rest = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]
experienced_meditation = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]

# Define the updated conditions
conditions = {
    'novice_rest_vs_experienced_rest': (novice_rest, experienced_rest),
    'novice_meditation_vs_experienced_meditation': (novice_meditation, experienced_meditation),
    'experienced_rest_vs_experienced_meditation': (experienced_rest, experienced_meditation),
    'experienced_meditation_vs_novice_meditation': (experienced_meditation, novice_meditation),
}

# Perform feature selection for each condition
significant_features = {}

# Iterate over each condition
for condition, (group1, group2) in conditions.items():
    t_stats = []
    p_values = []
    # Perform t-tests for each feature
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            t_stat, p_value = ttest_ind(group1[feature], group2[feature])
            t_stats.append(t_stat)
            p_values.append(p_value)
    
    # Adjust p-values for multiple comparisons
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    significant_features[condition] = {'significant_features': [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject[i] and scaled_data.columns[i] not in ['Level', 'EEG_State']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition
for condition in significant_features:
    print(f"Significant features for {condition}: {significant_features[condition]['significant_features']}")


# In[1]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)]
novice_meditation = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]
experienced_rest = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]
experienced_meditation = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]

# Define the updated conditions
conditions = {
    'novice_rest_vs_experienced_rest': (novice_rest, experienced_rest),
    'novice_meditation_vs_experienced_meditation': (novice_meditation, novice_rest),
    'experienced_rest_vs_experienced_meditation': (experienced_rest, experienced_meditation),
    'novice_rest_vs_novice_meditation': (novice_rest, novice_meditation),
}

# Perform feature selection for each condition
significant_features = {}

# Iterate over each condition
for condition, (group1, group2) in conditions.items():
    t_stats = []
    p_values = []
    # Perform t-tests for each feature
    for feature in scaled_data.columns:
        # Exclude 'Level' and 'EEG_State'
        if feature not in ['Level', 'EEG_State']:
            t_stat, p_value = ttest_ind(group1[feature], group2[feature])
            t_stats.append(t_stat)
            p_values.append(p_value)
    
    # Adjust p-values for multiple comparisons
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    # Filter out 'Level' and 'EEG_State' from significant features
    significant_features[condition] = {'significant_features': [scaled_data.columns[i] for i in range(len(reject)) if reject[i] and scaled_data.columns[i] not in ['Level', 'EEG_State']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition
for condition in significant_features:
    print(f"Significant features for {condition}: {significant_features[condition]['significant_features']}")


# In[13]:


print("Length of reject array:", len(reject))
print("Number of columns in the dataset:", len(scaled_data.columns))


# In[2]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define unique filenames
unique_filenames = filenames.unique()




print("Encoded values in EEG_State:", data['EEG_State'].unique())
print("Data type of EEG_State column:", data['EEG_State'].dtype)


# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform feature selection for each condition and each filename
significant_features = {}

# Iterate over each condition
for condition, ((level1, state1), (level2, state2)) in conditions.items():
    significant_features[condition] = {}
    
    # Iterate over each unique filename
    for filename in unique_filenames:
        # Select data for the current filename
        subset_data = scaled_data[filenames == filename]
        
        # Split the subset data into groups based on conditions
        group1 = subset_data[(data['Level'] == level1) & (data['EEG_State'] == state1)]
        group2 = subset_data[(data['Level'] == level2) & (data['EEG_State'] == state2)]
        
        t_stats = []
        p_values = []
        # Perform t-tests for each feature
        for feature in data.columns:
            # Exclude 'Level', 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                t_stat, p_value = ttest_ind(group1[feature], group2[feature])
                t_stats.append(t_stat)
                p_values.append(p_value)
        
        # Adjust p-values for multiple comparisons
        reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
        # Filter out 'Level' and 'EEG_State' from significant features
        significant_features[condition][filename] = {'significant_features': [scaled_data.columns[i] for i in range(len(reject)) if reject[i] and data.columns[i] not in ['Level', 'EEG_State', 'Filename']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition and filename
for condition, filename_data in significant_features.items():
    print(f"Condition: {condition}")
    for filename, features_data in filename_data.items():
        print(f"Filename: {filename}")
        print(f"Significant features: {features_data['significant_features']}")
        print(f"Number of statistically significant features: {len(features_data['significant_features'])}")


# In[5]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Extract filenames
filenames = data['Filename']

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define unique filenames
unique_filenames = filenames.unique()

print("Encoded values in EEG_State:", data['EEG_State'].unique())
print("Data type of EEG_State column:", data['EEG_State'].dtype)

# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': ((0, 0), (1, 0)),
    'novice_meditation_vs_experienced_meditation': ((0, 1), (1, 1)),
    'experienced_rest_vs_experienced_meditation': ((1, 0), (1, 1)),
    'experienced_meditation_vs_novice_meditation': ((1, 1), (0, 1)),
}

# Perform feature selection for each condition and each filename
significant_features = {}

# Iterate over each condition
for condition, ((level1, state1), (level2, state2)) in conditions.items():
    significant_features[condition] = {}
    
    # Iterate over each unique filename
    for filename in unique_filenames:
        # Select data for the current filename
        subset_data = scaled_data[filenames == filename]
        
        # Split the subset data into groups based on conditions
        group1 = subset_data[(data['Level'] == level1) & (data['EEG_State'] == state1)]
        group2 = subset_data[(data['Level'] == level2) & (data['EEG_State'] == state2)]
        
        # Perform Mann-Whitney U test for each feature
        u_stats = []
        p_values = []
        for feature in data.columns:
        # Exclude 'Level', 'EEG_State'
            if feature not in ['Level', 'EEG_State']:
                # Check if group1 and group2 have a non-zero size
                if len(group1) > 0 and len(group2) > 0:
                    u_stat, p_value = mannwhitneyu(group1[feature], group2[feature])
                    u_stats.append(u_stat)
                    p_values.append(p_value)
                else:
                    # Handle the case where one or both groups have zero size
                    print(f"Warning: One or both groups have zero size for feature {feature}. Mann-Whitney U test cannot be performed.")

        
        # Adjust p-values for multiple comparisons
        reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
        # Filter out 'Level' and 'EEG_State' from significant features
        significant_features[condition][filename] = {'significant_features': [scaled_data.columns[i] for i in range(len(reject)) if reject[i] and data.columns[i] not in ['Level', 'EEG_State', 'Filename']], 'corrected_p_values': corrected_p_values}

# Report significant features for each condition and filename
for condition, filename_data in significant_features.items():
    print(f"Condition: {condition}")
    for filename, features_data in filename_data.items():
        print(f"Filename: {filename}")
        print(f"Significant features: {features_data['significant_features']}")
        print(f"Number of statistically significant features: {len(features_data['significant_features'])}")


# In[9]:


# Display the data types of each column
print(data.dtypes)

# Display the unique values in each column
for column in data.columns:
    print(f"Unique values in {column}: {data[column].unique()}")


# In[5]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.power import TTestIndPower

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "Filename" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Define conditions
conditions = {
    'novice_rest_vs_experienced_rest': (scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)], 
                                         scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]),
    'novice_rest_vs_novice_meditation': (scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)], 
                                          scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]),
    'experienced_rest_vs_experienced_meditation': (scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)], 
                                                   scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]),
    'experienced_meditation_vs_novice_meditation': (scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)], 
                                                     scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]),
}

# Perform t-tests and compute effect sizes for each condition
effect_sizes = {}
for condition, (group1, group2) in conditions.items():
    t_stat, p_value = ttest_ind(group1, group2)
    effect_sizes[condition] = abs((group1.mean() - group2.mean()) / group1.std()).iloc[0]

# Define parameters for sample size calculation
alpha = 0.05  # Significance level
power = 0.90  # Desired power

# Create a TTestIndPower instance
analysis = TTestIndPower()

# Calculate sample size for each condition using effect sizes
sample_sizes = {}
for condition, effect_size in effect_sizes.items():
    sample_size = analysis.solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')
    sample_sizes[condition] = sample_size

# Print results
for condition, sample_size in sample_sizes.items():
    print(f"Required sample size for {condition}: {sample_size}")


# In[18]:


from sklearn.preprocessing import MinMaxScaler
import pandas as pd
# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "File_name", 'Level', and 'EEG_State' columns before scaling
data.drop(columns=["Filename", 'Level', 'EEG_State'], inplace=True)

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 'Novice') & (scaled_data['EEG_State'] == 'Rest')]
novice_meditation = scaled_data[(scaled_data['Level'] == 'Novice') & (scaled_data['EEG_State'] == 'Meditation')]
experienced_rest = scaled_data[(scaled_data['Level'] == 'Experienced') & (scaled_data['EEG_State'] == 'Rest')]
experienced_meditation = scaled_data[(scaled_data['Level'] == 'Experienced') & (scaled_data['EEG_State'] == 'Meditation')]

# Perform t-tests for each condition
results = {}

conditions = {
    'novice_rest': novice_rest,
    'novice_meditation': novice_meditation,
    'experienced_rest': experienced_rest,
    'experienced_meditation': experienced_meditation,
}

for condition, group in conditions.items():
    t_stats = []
    p_values = []
    
    for feature in scaled_data.columns:
        t_stat, p_value = ttest_ind(group[feature], scaled_data[feature])
        t_stats.append(t_stat)
        p_values.append(p_value)
    
    results[condition] = {'t_stats': t_stats, 'p_values': p_values}

# Adjust p-values for multiple comparisons
corrected_results = {}
for condition, result in results.items():
    p_values = result['p_values']
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    significant_features = [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject[i]]
    corrected_results[condition] = {'significant_features': significant_features, 'corrected_p_values': corrected_p_values}

# Report significant features for each condition
for condition in corrected_results:
    print(f"Significant features for {condition}: {corrected_results[condition]['significant_features']}")


# In[7]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn.feature_selection import SelectKBest, f_regression

# Load your dataset
data = pd.read_csv('Frameworkfeatures2-Copy1.csv')

# Drop the "File_name" column before scaling
data.drop(columns=["Filename"], inplace=True)

# Define LabelEncoder to transform string labels to integers
encoder = LabelEncoder()
data['Level'] = encoder.fit_transform(data['Level'])
data['EEG_State'] = encoder.fit_transform(data['EEG_State'])

# Scale the entire dataset
scaler = MinMaxScaler()
scaled_data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Split scaled dataset into subsets based on activity type and experience level
novice_rest = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 0)]
novice_meditation = scaled_data[(scaled_data['Level'] == 0) & (scaled_data['EEG_State'] == 1)]
experienced_rest = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 0)]
experienced_meditation = scaled_data[(scaled_data['Level'] == 1) & (scaled_data['EEG_State'] == 1)]

# Perform t-tests for each condition
results = {}


# Perform feature selection for each condition and each target variable
significant_features = {}

# Define the updated conditions
conditions = {
    'novice_rest_vs_experienced_rest': (novice_rest, experienced_rest),
    'novice_meditation_vs_experienced_meditation': (novice_meditation, experienced_meditation),
    'experienced_rest_vs_experienced_meditation': (experienced_rest, experienced_meditation),
    'experienced_meditation_vs_novice_meditation': (experienced_meditation, novice_meditation),
}

# Iterate over each condition
for condition, (group1, group2) in conditions.items():
    # Perform t-tests for EEG_State
    t_stats_eeg_state = []
    p_values_eeg_state = []
    for feature in scaled_data.columns:
        t_stat, p_value = ttest_ind(group1[feature], group2[feature])
        t_stats_eeg_state.append(t_stat)
        p_values_eeg_state.append(p_value)
    
    # Perform t-tests for Level
    t_stats_level = []
    p_values_level = []
    for feature in scaled_data.columns:
        t_stat, p_value = ttest_ind(group1[feature], group2[feature])
        t_stats_level.append(t_stat)
        p_values_level.append(p_value)
    
    # Adjust p-values for multiple comparisons for EEG_State
    reject_eeg_state, corrected_p_values_eeg_state, _, _ = multipletests(p_values_eeg_state, method='bonferroni')
    significant_features_eeg_state = [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject_eeg_state[i]]
    
    # Adjust p-values for multiple comparisons for Level
    reject_level, corrected_p_values_level, _, _ = multipletests(p_values_level, method='bonferroni')
    significant_features_level = [scaled_data.columns[i] for i in range(len(scaled_data.columns)) if reject_level[i]]
    
    # Store the significant features for both EEG_State and Level
    significant_features[condition] = {'significant_features_eeg_state': significant_features_eeg_state, 
                                       'corrected_p_values_eeg_state': corrected_p_values_eeg_state,
                                       'significant_features_level': significant_features_level, 
                                       'corrected_p_values_level': corrected_p_values_level}

# Report significant features for each condition and each target variable
for condition in significant_features:
    print(f"Significant features for {condition} (EEG_State): {significant_features[condition]['significant_features_eeg_state']}")
    print(f"Significant features for {condition} (Level): {significant_features[condition]['significant_features_level']}")


# In[5]:


from sklearn.feature_selection import SelectKBest, f_classif

# Create a dictionary to hold the significant features for each subset
significant_features = {}

# Perform feature selection for each subset
for subset, subset_data in subsets.items():
    # Remove 'Level' and 'EEG_State' columns if they exist
    if 'Level' in subset_data.columns:
        subset_data.drop(columns=['Level'], inplace=True)
    if 'EEG_State' in subset_data.columns:
        subset_data.drop(columns=['EEG_State'], inplace=True)
    
    # Perform feature selection using SelectKBest with ANOVA F-value as the scoring function
    selector = SelectKBest(score_func=f_classif, k='all')
    selector.fit(subset_data.drop(columns=['Target']), subset_data['Target'])
    
    # Get the indices of the top k features
    top_indices = selector.scores_.argsort()[::-1]
    
    # Get the names of the top k features
    top_features = subset_data.drop(columns=['Target']).columns[top_indices]
    
    # Store the significant features for the subset
    significant_features[subset] = top_features

# Print the significant features for each subset
for subset, features in significant_features.items():
    print(f"Significant features for {subset}: {features}")


# In[ ]:


from scipy.stats import ttest_ind

# Perform t-tests for each feature for different conditions
conditions = ['novice_rest', 'experienced_rest', 'novice_meditation', 'experienced_meditation']
results = {}

for condition in conditions:
    group1, group2 = condition.split('_')
    t_stats = []
    p_values = []
    
    for feature in data.columns:
        t_stat, p_value = ttest_ind(data[(data['Level'] == 'Novice') & (data['EEG_State'] == group1)][feature],
                                     data[(data['Level'] == 'Experienced') & (data['EEG_State'] == group2)][feature])
        t_stats.append(t_stat)
        p_values.append(p_value)
    
    results[condition] = {'t_stats': t_stats, 'p_values': p_values}

# Adjust p-values for multiple comparisons
corrected_results = {}
for condition in results:
    p_values = results[condition]['p_values']
    reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')
    significant_features = [data.columns[i] for i in range(len(data.columns)) if reject[i]]
    corrected_results[condition] = {'significant_features': significant_features, 'corrected_p_values': corrected_p_values}

# Report significant features for each condition
for condition in corrected_results:
    print(f"Significant features for {condition}: {corrected_results[condition]['significant_features']}")


# In[ ]:


from sklearn.preprocessing import MinMaxScaler

# Scale the data
scaler = MinMaxScaler()
scaled_novice_rest = pd.DataFrame(scaler.fit_transform(novice_rest), columns=novice_rest.columns)
scaled_experienced_rest = pd.DataFrame(scaler.fit_transform(experienced_rest), columns=experienced_rest.columns)

# Perform t-tests for each feature
results = []
for feature in data.columns:
    t_stat, p_value = ttest_ind(scaled_novice_rest[feature], scaled_experienced_rest[feature])
    results.append((feature, t_stat, p_value))

# Adjust p-values for multiple comparisons
p_values = [p_value for _, _, p_value in results]
reject, corrected_p_values, _, _ = multipletests(p_values, method='bonferroni')

# Report significant features
significant_features = [results[i][0] for i in range(len(results)) if reject[i]]
print("Significant features:", significant_features)


# In[ ]:




