#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt


file_names = ['c_med_2.txt', 'c_med_6.txt', 'c_med_9.txt', 'c_med_12.txt', 'c_med_13.txt', 'c_med_19.txt', 'c_med_22.txt', 'c_med_26.txt', 'c_med_n12.txt']
c_rest = ['c_rest_2.txt', 'c_rest_6.txt', 'c_rest_9.txt', 'c_rest_12.txt', 'c_rest_13.txt', 'c_rest_19.txt', 'c_rest_22.txt', 'c_rest_26.txt', 'c_rest_n12.txt']
m_rest = ['m_rest_7.txt', 'm_rest_9.txt', 'm_rest_14.txt', 'm_rest_20.txt', 'm_rest_24.txt', 'm_rest_25.txt', 'm_rest_31.txt']
m_med = ['m_med_7.txt', 'm_med_9.txt', 'm_med_14.txt', 'm_med_20.txt', 'm_med_24.txt', 'm_med_25.txt', 'm_med_31.txt']


# In[12]:


def read_file_and_calculate_mean(file_name):
    try:
        df = pd.read_csv(file_name, sep=',', encoding='latin-1', on_bad_lines='skip')
    except UnicodeDecodeError:
        df = pd.read_csv(file_name, sep=',', encoding='ISO-8859-1', on_bad_lines='skip')
    
    mean_df = df.mean()
    return mean_df


# In[13]:


c_med_means = [read_file_and_calculate_mean(file_name) for file_name in file_names]
c_rest_means = [read_file_and_calculate_mean(file_name) for file_name in c_rest]
m_rest_means = [read_file_and_calculate_mean(file_name) for file_name in m_rest]
m_med_means = [read_file_and_calculate_mean(file_name) for file_name in m_med]


# In[14]:


all_means_df = pd.DataFrame(c_med_means, index=file_names)
all_means_df1 = pd.DataFrame(c_rest_means, index=c_rest)
all_means_df2 = pd.DataFrame(m_rest_means, index=m_rest)
all_means_df3 = pd.DataFrame(m_med_means, index=m_med)


# In[15]:


print("All Controls at Meditation Means:")
print(all_means_df)


# In[16]:


print("All Controls at Rest Means:")
print(all_means_df1)


# In[17]:


print("All Meditators at Rest Means:")
print(all_means_df2)


# In[18]:


print("All Meditators at Meditation Means:")
print(all_means_df3)


# In[24]:


plt.figure(figsize=(10, 6))
plt.plot(all_means_df.mean(), marker='o', linestyle='-', color='red', label='Controls at Meditation')
plt.plot(all_means_df3.mean(), marker='o', linestyle='-', color='blue', label='Meditators at Meditation')
plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Comparative Multilinear Plot of Mean Values for Controls vs Meditators at Meditation')
plt.legend()
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
plt.plot(all_means_df2.mean(), marker='o', linestyle='-', color='blue', label='Meditators at Rest')
plt.plot(all_means_df1.mean(), marker='o', linestyle='-', color='red', label='Controls at Rest')
plt.xlabel('Columns')
plt.ylabel('Mean Value')
plt.title('Comparative Multilinear Plot of Mean Values for Controls vs Meditators at Rest')
plt.legend()
plt.show()


# In[ ]:




