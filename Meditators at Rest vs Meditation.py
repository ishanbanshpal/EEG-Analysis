#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


m_rest = ['m_rest_7.txt', 'm_rest_9.txt', 'm_rest_14.txt', 'm_rest_20.txt', 'm_rest_24.txt', 'm_rest_25.txt', 'm_rest_31.txt'] 


def read_file_and_calculate_stats(file_name):
    try:
        df = pd.read_csv(file_name, sep=',', encoding='latin-1', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, sep=',', encoding='ISO-8859-1',on_bad_lines='skip')
    mean_df = df.mean()
    median_df = df.median()
    std_df = df.std()
    return mean_df, median_df, std_df

mean_dfs = []
median_dfs = []
std_dfs = []

for file_name in m_rest:
    mean_df, median_df, std_df = read_file_and_calculate_stats(file_name)
    mean_dfs.append(mean_df)
    median_dfs.append(median_df)
    std_dfs.append(std_df)

all_means_df2 = pd.concat(mean_dfs, axis=1)
all_means_df2.columns = m_rest

all_median_df2 = pd.concat(median_dfs, axis=1)
all_median_df2.columns = m_rest

all_std_df2 = pd.concat(std_dfs, axis=1)
all_std_df2.columns = m_rest


# In[31]:


for file_name in m_rest:
    df = pd.read_csv(file_name, sep=',', encoding='latin-1')
    df = df.reindex(columns=df.columns)
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        plt.violinplot(df[column], showmeans=True, showmedians=True)
        plt.xlabel('Columns')
        plt.ylabel('Value')
        plt.title(f'Violin Plot of {column} for {file_name}')
        plt.show()


# In[5]:


m_med = ['m_med_7.txt', 'm_med_9.txt', 'm_med_14.txt', 'm_med_20.txt', 'm_med_24.txt', 'm_med_25.txt', 'm_med_31.txt'] 


def read_file_and_calculate_stats(file_name):
    try:
        df = pd.read_csv(file_name, sep=',', encoding='latin-1', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, sep=',', encoding='ISO-8859-1',on_bad_lines='skip')
    mean_df = df.mean()
    median_df = df.median()
    std_df = df.std()
    return mean_df, median_df, std_df

mean_dfs = []
median_dfs = []
std_dfs = []

for file_name in m_med:
    mean_df, median_df, std_df = read_file_and_calculate_stats(file_name)
    mean_dfs.append(mean_df)
    median_dfs.append(median_df)
    std_dfs.append(std_df)

all_means_df3 = pd.concat(mean_dfs, axis=1)
all_means_df3.columns = m_med

all_median_df3 = pd.concat(median_dfs, axis=1)
all_median_df3.columns = m_med

all_std_df3 = pd.concat(std_dfs, axis=1)
all_std_df3.columns = m_med




# In[30]:


for file_name in m_med:
    df = pd.read_csv(file_name, sep=',', encoding='latin-1')
    df = df.reindex(columns=df.columns) 
    for column in df.columns:
        plt.figure(figsize=(8, 6))
        plt.violinplot(df[column], showmeans=True, showmedians=True)
        plt.xlabel('Columns')
        plt.ylabel('Value')
        plt.title(f'Violin Plot of {column} for {file_name}')
        plt.show()


# In[13]:


paired_files = zip(m_rest,m_med)

for file_name, m_med_file in paired_files:
    plt.figure(figsize=(10, 6))
    plt.plot(all_means_df2[file_name], marker='o', linestyle='-',color='orange', label='Meditators in Rest')
    plt.plot(all_means_df3[m_med_file], marker='o', linestyle='-',color='blue', label='Meditators in Meditation')
    plt.xlabel('Columns')
    plt.ylabel('Mean Value')
    plt.title(f'Comparative Line Plot of Mean Values for {file_name} and {m_med_file}')
    plt.legend(title='Group')
    plt.show()


# In[14]:


paired_files = zip(m_rest, m_med)

for file_name, m_med_file in paired_files:
    plt.figure(figsize=(10, 6))
    all_median_df2[file_name].plot(kind='bar', figsize=(10, 6), alpha=0.7, color='orange', label='Meditators in Rest')
    all_median_df3[m_med_file].plot(kind='bar', figsize=(10, 6), alpha=0.7, color='blue', label='Meditators in Meditation')
    plt.xlabel('Columns')
    plt.ylabel('Median Value')
    plt.title(f'Comparative Bar Plot of Median Values for {file_name} and {m_med_file}')
    plt.legend(title='Group')
    plt.show()


# In[26]:


paired_files = zip(m_rest, m_med)
plt.figure(figsize=(10, 6))

for file_name, m_med_file in paired_files:
    plt.plot(all_means_df2[file_name], marker='o', linestyle='-', color='orange', label=f'Meditators in Rest - {file_name}')
    plt.plot(all_means_df3[m_med_file], marker='o', linestyle='-', color='blue', label=f'Meditators in Meditation - {m_med_file}')

plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Comparative Multilinear Plot of Mean Values for Meditators')
plt.legend([f'Controls in Rest - {m_med_file}', f'Controls in Meditation - {file_name}'])
plt.show()




# In[29]:


plt.figure(figsize=(10, 6))

plt.plot(all_means_df2.mean(), marker='o', linestyle='-', color='orange', label='Meditators in Rest')
plt.plot(all_means_df3.mean(), marker='o', linestyle='-', color='blue', label='Meditators in Meditation')

plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Comparative Multilinear Plot of Mean of all Columns for Meditators')
plt.legend(title='Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[32]:


print("Dataframe of Mean Values of Meditators at Rest:")
display(all_means_df2)


# In[33]:


print("Dataframe of Mean Values of Meditators at Meditation:")
display(all_means_df3)


# In[34]:


print("Dataframe of Median Values of Meditators at Rest:")
display(all_median_df2)


# In[35]:


print("Dataframe of Mean Values of Meditators at Meditation:")
display(all_median_df3)


# In[ ]:




