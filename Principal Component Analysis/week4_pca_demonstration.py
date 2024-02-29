#!/usr/bin/env python
# coding: utf-8

# # Demonstration of PCA

# Please note that this notebook is not assessed. We will touch on topics that we do not cover in this specialization, including dataframes, machine learning algorithms, etc. While we encourage you to go through this notebook as it will give you an idea about how PCA is used in real life, feel free to skip this notebook if you are unable to follow it.

# ## Learning objectives
# 
# 1. Understanding and interpreting the correlation between different features of a dataset.
# 2. Applying PCA on a real world dataset and understanding how much data is stored in which components.
# 3. Visualizing high dimensional data by first reducing it to two dimensions by using PCA and then plotting it.
# 4. Observing how the performance of a model varies with the number of principle components used.

# Two of the uses of PCA are to visualize high dimensional datasets and improve the training speed of machine learning models. In this notebook, we will demonstrate how PCA helps us with both of these.

# We will analyze the breast cancer dataset of sklearn. We will first load the dataset and split it into the training and test sets. Then, we will perform logistic regression on the training set and evaluate our model on the test set. Having done that, we will visualize and analyze correlations in the data. We would explore how PCA helps us compress the data and project it onto two dimensions so that we can plot it. Lastly, we will perform logistic regression on the compressed data and see how that stacks up against the performance of the model on the original data.

# Let's first import the packages we need.

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

get_ipython().run_line_magic('matplotlib', 'inline')


# We will use the breast cancer dataset of sklearn in this notebook. This dataset contains information about malignant and benign tumours. We aim to create a model which can classify a tumour as being malignant or benign.

# ## Loading The Dataset

# To start off, we will load the dataset and print a short description about it. You are encouraged to read the description before moving ahead.

# In[ ]:


dat = datasets.load_breast_cancer()
print(dat.DESCR)


# We will now load the data into a dataframe and print the first 5 entries. Can you guess how some of the these attributes might be related? For example, could mean radius and mean perimeter be linked?

# In[ ]:


df_all = pd.DataFrame(dat['data'], columns=list(dat['feature_names']))
df_all.head()


# ## Creating Train and Test Sets

# In order to enable our model to make predictions on whether a tumour is malignant or benign, we need to train it from some data. However, to then evaluate the performance of our model, we need to check that it classifies unseen data correctly. That is why we split the data that we have into two sets - the training set and the test set. We train the parameters for the model by using the training set and then evaluate how good is the model using the test set.
# 
# We take 70% of the samples into the training set and 30% into the test set, which is conventional for the size of the dataset that we have.

# In[ ]:


# We do a 70/30 split
TEST_SIZE_RATIO = 0.3

# Setting up X and y
X = df_all
y = pd.Series(list(dat['target']))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE_RATIO, random_state=0)

# Normalizing the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("X_train.shape, y_train.shape", X_train.shape, y_train.shape)
print("X_test.shape, y_test.shape", X_test.shape, y_test.shape)


# ## Performing Logistic Regression

# We will now perform Logistic Regression on the unmodified data. For those of you who do not know about Logistic Regression, it is a technique which lets us classify data into different classes. For example, we can get the image of a fruit and try to classify if it is an orange or an apple. In that case the classes are orange and apple.
# 
# In this case, the classes are malignant (represented by the number 0) and benign (represented by the number 1). Based on the data, we would try to classify a tumour as being malignant or benign.
# 
# In order to evaluate the performance of the model, we will use a metric called the `f1 score`, which assesses how well the model is doing in predicting whether a tumour is malignant or benign and gives a numerical score based on that. We aim to get the f1 score to be as high as possible.

# In[ ]:


model = LogisticRegression(random_state=0).fit(X_train, y_train)
print("Training score: ", f1_score(y_train, model.predict(X_train)))
print("Testing score: ", f1_score(y_test, model.predict(X_test)))


# When we train the model on the initial (normalized) data, we get a test set score of 98.17%.

# ## Visualizing Correlation of the Features

# We would expect some features to be highly correlated to one another. Let us create a heatmap which will let us visualize the correlation between the different features.
# 
# The code for the following function has been taken from [this](https://colab.research.google.com/github/gal-a/blog/blob/master/docs/notebooks/sklearn/sklearn_logistic_regression_vs_gbm.ipynb#scrollTo=ayp_TDIq6qJj) notebook.

# In[ ]:


def correlation_matrix(y, X, is_plot=False):
  # Calculate and plot the correlation symmetrical matrix
  # Return:
  # yX - concatenated data
  # yX_corr - correlation matrix, pearson correlation of values from -1 to +1
  # yX_abs_corr - correlation matrix, absolute values
  
  yX = pd.concat([y, X], axis=1)
  yX = yX.rename(columns={0: 'TARGET'})  # rename first column

  print("Function correlation_matrix: X.shape, y.shape, yX.shape:", X.shape, y.shape, yX.shape)
  print()

  # Get feature correlations and transform to dataframe
  yX_corr = yX.corr(method='pearson')

  # Convert to abolute values
  yX_abs_corr = np.abs(yX_corr) 
  
  if is_plot:
    plt.figure(figsize=(10, 10))
    plt.imshow(yX_abs_corr, cmap='RdYlGn', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(yX_abs_corr)), yX_abs_corr.columns, rotation='vertical')
    plt.yticks(range(len(yX_abs_corr)), yX_abs_corr.columns);
    plt.suptitle('Pearson Correlation Heat Map (absolute values)', fontsize=15, fontweight='bold')
    plt.show()
  
  return yX, yX_corr, yX_abs_corr

# Build the correlation matrix for the train data
yX, yX_corr, yX_abs_corr = correlation_matrix(y, X, is_plot=True) 


# The `TARGET` label above refers to whether the tumour is malignant(0) or benign(1). Green cells indicate very high correlation whereas red cells indicate very low correlation.
# 
# From the above graph, we can see that `mean radius` and `mean perimeter` are very strongly correlated to one another (correlation is almost equal to 1). `mean radius` and `mean area` are very strongly related to one another as well. However, `worst fractal dimension` and `mean radius` are completely unrelated to one another (correlation is almost equal to 0). Can you identify some other features which are very strongly related or unrelated to one another?
# 
# We can drop some of these features and combine the others and still lose only minimal amount of information. This is what we will do in the next section when we apply PCA to the data.

# ## Performing PCA on the Dataset

# We will now perform PCA on our training set. This can be done trivially using the sklearn library. Note that the eigenbasis is calculated using only the data points in the training set. This is because the model should perform well on unseen data as well. If we include data about the test set when we find the eigenbasis, then our model might get biased and the test set would no longer provide an unbiased judgement of our model.
# 
# We have already normalized the data before and hence we do not need to normalize it again.

# In[ ]:


# Applying PCA
pca = PCA()
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

# Plotting the amount of information stored in each component
plt.ylabel('Variance')
plt.xlabel('Component Number')
plt.bar(np.arange(30) + 1, pca.explained_variance_ratio_)
plt.show()


# As you can see, the amount of information stored in a component is really high for the first few dimensions, but then it falls down very rapidly. Only 1.4% of the variance is stored in the 7th component. Moreover, it falls to a completely insignificant 0.25% in the 16th component. We retain alomst 85% of the variance of the original data by using only 5 components, which is one-sixth of the number of features of the original data.
# 
# We are printing the numerical values of the variance stored in each component below, should you wish to view them.

# In[ ]:


print(pca.explained_variance_ratio_)


# ## Visualizing the Data

# One of the uses of PCA is that it lets us visualize high dimensional data.
# 
# PCA lets us project our original data into a two dimensional space. This lets us visualize the data by plotting it on a graph. Below, we will visualize the training set by performing PCA on it and only considering the first 2 components.

# In[ ]:


pca = PCA(n_components=2)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

X_train_reduced_malignant = np.array([x for x, y in zip(X_train_reduced, y_train) if y == 0])
X_train_reduced_benign = np.array([x for x, y in zip(X_train_reduced, y_train) if y == 1])

plt.scatter(*X_train_reduced_malignant.T, color='red')
plt.scatter(*X_train_reduced_benign.T, color='blue')
plt.title('Training Set After PCA')
plt.legend(['malignant', 'benign'])
plt.xlabel('Coordinate of first principle component')
plt.ylabel('Coordinate of second principle component')
plt.show()


# ## Performing Logistic Regression

# We will now perform logistic regression again, but this time we will first perform PCA on the data and only consider the first 5 components.

# In[ ]:


pca = PCA(n_components=5)
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)

model = LogisticRegression(random_state=0).fit(X_train_reduced, y_train)
print("Training score: ", f1_score(y_train, model.predict(X_train_reduced)))
print("Testing score: ", f1_score(y_test, model.predict(X_test_reduced)))


# This time we get a test set score of 97.7%, which is only marginally worse than the original score of 98.17%. However, by using only one-sixth of the amount of data we had initially used, our model shall be trained much faster. While the effect might not be noticeable in this case, it can be huge in certain applications, where the models might have taken many days to train otherwise.

# We encourage you to vary the value of `n_components` in the code above and see how the test set accuracy changes. In particular, notice that by using just a single component, we already get a score of 91.59%. Moreover, if we use 14 components, we get a score of 98.17%, which is equal to what we got for the original data, but uses less than half the number of features.

# We plot the training score and test score against the number of components. Note that while the training score and test score generally increase with an increase in the number of components, this does not always have to hold. In practice, you need to experiment with different values of `n_components` and find out what works for the specific problem that you are trying to solve.

# In[ ]:


X = np.arange(30) + 1
Y = []

for i in X:
    pca = PCA(n_components=i)
    X_train_reduced = pca.fit_transform(X_train)
    model = LogisticRegression(random_state=0).fit(X_train_reduced, y_train)
    Y.append(f1_score(y_train, model.predict(X_train_reduced)))

plt.plot(X, Y)
plt.xlabel('Number of Components')
plt.ylabel('Training Score')
plt.show()

Y = []

for i in X:
    pca = PCA(n_components=i)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    model = LogisticRegression(random_state=0).fit(X_train_reduced, y_train)
    Y.append(f1_score(y_test, model.predict(X_test_reduced)))

plt.plot(X, Y)
plt.xlabel('Number of Components')
plt.ylabel('Test Score')
plt.show()

