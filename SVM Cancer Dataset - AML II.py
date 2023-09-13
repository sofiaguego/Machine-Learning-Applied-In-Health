#!/usr/bin/env python
# coding: utf-8

# # Support Vector Machine (SVM) Classifier - Breast Cancer Dataset
# 

# The purpose of this notebook is to use and compare different classification algorithms and evaluate its accuracy (in this case SVM) to classify either if the type of Breast Cancer is malignant or bening. The dataset was taken from SkLearn. 

# In[51]:


from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
from time import perf_counter, sleep


# In[52]:


#https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
cancer = datasets.load_breast_cancer()
print(f"Features: {cancer.feature_names}")
print("")
print(f"Target {cancer.target_names}")
      


# In[53]:


df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df.head(10)


# In[54]:


df.shape


# **0 means malign
# 1 means benign**

# In[55]:


df["target"] = cancer.target
df.head()


# In[56]:


df.describe()


# # **Checking for missing values:**

# In[74]:


df.isna().sum()


# **We do not need to scale in SVM**

# **DATA SPLIT:** 

# In[75]:


X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Train: {X_train.shape}")
print(f"Test:{y_test.shape}")


# **MODELLING/TRAINING:**

# In[76]:


start = perf_counter()

model = SVC(kernel="linear") #HERE WE TRY DIFFERENT KERNELS: rbf, sigmoid, poly
model.fit(X_train, y_train)

end = perf_counter()
print(f"Time taken to execute code : {end-start}")


# # **Evaluation: Accuracy and Confusion Matrix**

# In[77]:


y_predicted = model.predict(X_test)



print(f"Accuracy: {metrics.accuracy_score(y_test, y_predicted)}")


# Usually, rbf is the best. If we wanna do this for NN, we just need to scale. 

# In[78]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score


# In[79]:


cm = metrics.confusion_matrix(y_test, y_predicted, labels=model.classes_)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()


# Understanding the Confusion Matrix: 
# - If we check the numbers 59 VS 106 we see that the data is not balanced which is not "ideal". However, 2 and 4 is the error and it is a good error. 
#     

# In[ ]:




