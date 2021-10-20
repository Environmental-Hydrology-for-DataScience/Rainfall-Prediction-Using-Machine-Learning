# Installing Packages

```python
!pip install pandas
```

```python
!pip install seaborn
```

```python
!pip install numpy
```

```python
!pip install matplotlib
```

```python
!pip install sklearn
```

```python
!pip install warnings
```

# Importing Packages

```python
import pandas as pd
```

```python
import seaborn as sns
```

```python
import numpy as np
```

```python
import matplotlib.pyplot as plt
```

```python
import sklearn
```

```python
import warnings
```

# Reading Dataset

```python
df = pd.read_csv("Dataset_Name.csv")
```
# Exploring and Visualizing Data

```python
df.head()
```
By default it returns the first 5 rows of dataset.

```python
df.shape
```
It is used to know the dimensions(No.of rows ,No.of columns) of Dataset.

```python
df.info()
```
This method is used to know the summary of DataFrame .The Information like Columnname ,Column Datatypes, Non-null values and Memory usage.

```python
df.describe()
```
This method is used to generate descriptive statistics that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.

```python
df.isna().sum()
```
This method is used to detect missing values for an array-like object and count them for every column.

```pyton
df["Column_Name"].value_counts()
```
It returns a Series containing counts of unique values

```python
sns.barplot(x = None, y = None, data = None)
```
A bar plot represents an estimate of central tendency for a numeric variable with the height of each rectangle and provides some indication of the uncertainty around that estimate using error bars. 

```python
sns.distplot(a = None, bins = None)
```
It is used basically for univariant set of observations and visualizes it through a histogram.

```python
sns.lineplot(x = None, y = None, data = None)
```
It draws a line plot with the possibility of several semantic groupings. The relationship between x and y can be shown for different subsets of the data using the hue, size, and style parameters

```python
sns.jointplot(x = None, y = None, data = None)
```
It draws a plot of two variables with bivariate and univariate graphs.

```python
sns.heatmap(data = None)
```
This gives a graphical representation of data using colors to visualize the value of the matrix.

# Cleaning and Pre-processing Data

```python
sklearn.utils.resample(*arrays, replace=True, n_samples=None, random_state=None)
```
This function resamples arrays or sparse matrices in a consistent way. The default strategy implements one step of the bootstrapping procedure.

```python
DataFrame.iloc()
```
Purely integer-location based indexing for selection by position.

```python
sklearn.impute.SimpleImputer(missing_values = np.nan, strategy = "most_frequent")
```
This replaces missing values using the most frequent value along each column.

```python
sklearn.preprocessing.LabelEncoder()
```
It is a utility class to help normalize labels such that they contain only values between 0 and n_classes-1

```python
sklearn.preprocessing.StandardScaler()
```
This helps in standardizing features by removing the mean and scaling to unit variance.

# Modelling

```python
sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None)
```
This split arrays or matrices into random train and test subsets

```python
sklearn.linear_model.LogisticRegression()
```
In statistics, the logistic model (or logit model) is used to model the probability of a certain class or event existing such as pass/fail, win/lose, alive/dead or healthy/sick. This can be extended to model several classes of events such as determining whether an image contains a cat, dog, lion, etc. Each object being detected in the image would be assigned a probability between 0 and 1, with a sum of one. Logistic regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable, although many more complex extensions exist. In regression analysis, logistic regression (or logit regression) is estimating the parameters of a logistic model.

```python
sklearn.tree.DecisionTreeClassifier()
```
Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree. We can represent any boolean function on discrete attributes using the decision tree.

```python
sklearn.ensemble.RandomForestClassifier()
```
A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 

```python
sklearn.lightgbm.LGBMClassifier()
```
Light GBM is a gradient boosting framework that uses tree based learning algorithm. Light GBM grows tree vertically while other algorithm grows trees horizontally meaning that Light GBM grows tree leaf-wise while other algorithm grows level-wise. It will choose the leaf with max delta loss to grow. When growing the same leaf, Leaf-wise algorithm can reduce more loss than a level-wise algorithm.

```python
sklearn.naive_bayes.GaussianNB()
```
The naive Bayes classifier assumes all the features are independent to each other. Even if the features depend on each other or upon the existence of the other features. A Gaussian Naive Bayes algorithm is a special type of NB algorithm. It’s specifically used when the features have continuous values. It’s also assumed that all the features are following a gaussian distribution i.e, normal distribution.

```python
sklearn.metrics.confusion_matrix(y_true, y_pred)
```
A confusion matrix is a table that is often used to describe the performance of a classification model (or "classifier") on a set of test data for which the true values are known.

# Algorithms

 | S.No | Name | Defination |
 |-----|-----|-----|
 | 1 | Logistic Regression|	Logistic regression is used to describe data and to explain the relationship between one dependent binary variable. |
 | 2 | Random Forest	| A Random forest is a machine learning algorithm that develops large numbers of random decision trees analyzing sets of variables. |
 | 3 | Decision Trees	| A decision tree is a map of the possible outcomes of a series of related choices. It allows to weigh possible actions against one another based on their probabilities. This can can be used to map out an decision that predicts the best choice mathematically. |
 | 4 | LightGBM	|  Light GBM is a fast, distributed, high-performance gradient boosting framework based on decision tree algorithm, used for ranking and classification.The tree is grows vertically. |
 | 5 | Naive Bayes	 | Naive Bayes is a collection of classification algorithms based on Bayes’ Theorem. It is not a single algorithm but a family of algorithms where all of them share a common principle, i.e. every pair of features being classified is independent of each other. |
