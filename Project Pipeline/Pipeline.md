* # Importing Packages
  
  Packages used:
  
  1. ### Numpy:
	  NumPy is the fundamental package for scientific computing in Python. It is a Python library that provides a multidimensional array object, various derived objects (such as masked arrays and matrices), and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, selecting, I/O, discrete Fourier transforms, basic linear algebra, basic statistical operations, random simulation and much more.
  
  2. ### Pandas:
	  Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real-world data analysis in Python. Additionally, it has the broader goal of becoming the most powerful and flexible open source data analysis/manipulation tool available in any language.
  
  3. ### Matplotlip:
	  Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib produces publication-quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shell, web application servers, and various graphical user interface toolkits.
  
  4. ### Seaborn:
	  Seaborn is a library for making statistical graphics in Python. It builds on top of matplotlib and integrates closely with pandas data structures. Seaborn helps you explore and understand your data. Its plotting functions operate on dataframes and arrays containing whole datasets and internally perform the necessary semantic mapping and statistical aggregation to produce informative plots. Its dataset-oriented, declarative API lets you focus on what the different elements of your plots mean, rather than on the details of how to draw them.
  
  5. ### Sklearn:
	  Scikit-learn (Sklearn) is the most useful and robust library for machine learning in Python. It provides a selection of efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction via a consistence interface in Python.

* # Loading Dataset:
  
  We have used Rain in Australia Dataset from Kaggle. We imported the dataset into our Google Colab in the following way:

    1. ### Step 1: Selecting the respective dataset from kaggle.
	      Selected the dataset of our choice and copied the URL.
 
    2. ### Step 2: Downloading API Credentials.
	      To download data from Kaggle, we needed to authenticate with the Kaggle services. For this purpose, we needed an API token. This token can be easily generated from the profile section of our Kaggle account. Simply, navigated to your Kaggle profile and then, Clicked the Account tab and then scrolled down to the API section and Clicked on "Create New API Token" button.
        
        A file named “kaggle.json” was downloaded which contains the username and the API key. 

    3. ### Step 3: Settig up the Colab Notebook.
	      Fired up a Google Colab notebook and connected it to the cloud instance. Then, uploaded the “kaggle.json” file that we just downloaded from Kaggle.
        
        Then we ran the following commands in the Colab Notebook:
          
          1. Install the Kaggle library
              ```python
              ! pip install kaggle
              ```

          2. Make a directory named “.kaggle”
              ```python
              ! mkdir ~/.kaggle
              ```

          3. Copy the “kaggle.json” into this new directory
              ```python
              ! cp kaggle.json ~/.kaggle/
              ```
              
          4. Allocate the required permission for this file.
              ```python
              ! chmod 600 ~/.kaggle/kaggle.json
              ```
              
     4. ### Step 4: Downloading dataset
        To download the dataset:
        
        ```python
        ! kaggle datasets download jsphyg/weather-dataset-rattle-package
        ```
            
        To unzip the downloaded file:
        
        ```python
        ! unzip weather-dataset-rattle-package
        ```
            
        A csv file(weatherAUS.csv) was created after this step.


     5. ### Step 5: Reading the csv file
        We created dataframe by reading the csv file to access the dataset with ease.
       

* # Exploratory Data Analysis
	
  Exploratory data analysis is an approach of analysing data sets to summarize their main characteristics, often using statistical graphics and other data visualization methods. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modelling or hypothesis testing task. EDA also helps to ﬁnd insights that were not evident or worth investigating to businessstakeholders and researchers.
  We performed EDA on our dataset and acquired some interesting insights.
 
  ![image](https://user-images.githubusercontent.com/89296568/140265661-1acac67d-ddef-4b8e-8560-f8389b2138ff.png)

  ![image](https://user-images.githubusercontent.com/89296568/140265682-b79c13a9-f088-4a87-ad46-3ca5e85cb004.png)

  ![image](https://user-images.githubusercontent.com/89296568/140265699-33995eff-aed5-4a28-947e-f4e66d3fb36a.png)

* # Pre-processing Data

  1. ### Dealing with Class Imbalance
    
      We have learned from our EDA that our dataset is highly imbalanced. Imbalanced data results in biased results as our model doesn’t learn much about the minority class. 
      So, we oversampled our dataset with respect to “RainTomorrow” attribute with the help of resample from sklearn.utils.
      
      ```python
      from sklearn.utils import resample
      no = df[df.RainTomorrow == 0]
      yes = df[df.RainTomorrow == 1]
      yes_oversampled = resample(yes, replace=True, n_samples=len(no), random_state=123)
      df = pd.concat([no, yes_oversampled])
      ```
      ![image](https://user-images.githubusercontent.com/89296568/140266513-042bc242-f696-44c7-9b38-333f32722c86.png)


  2. ### Feature Selection
      
      Feature selection is the process of reducing the number of input variables when developing a predictive model.
      We did manual feature selection based on the results of our EDA. We considered the attribute “RainTomorrow” as our dependent variable (Y) as it is what we have to predict. We have considered all the remaining attributes except “Date”, “Evaporation”, “Sunshine” as Independent Variables (X) because Date doesn’t effect our model and Evaporation and Sunshine have very high percentage of missing values.

      ```python
      X = df.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
      Y = df.iloc[:,-1].values
      ```
      
  3. ### Dealing with Missing Values
    
      We have seen in our EDA that many attributes contain high percentage of missing values which could result in bad accuracy of our model. We have used Simple Imputer from ski-kit learn to fill the missing values with most frequent values in respective columns.
      ```python
      from sklearn.impute import SimpleImputer
      imp = SimpleImputer(missing_values=np.nan,strategy='most_frequent') # Fill the missing values with Mode
      X = imp.fit_transform(X)
      Y = imp.fit_transform(Y)
      ```
      
  4. ### Encoding Categorical data
      
      Categorical feature is one that has two or more categories, but there is no intrinsic ordering to the categories. We have a few categorical features – Location, WindGustDir, WindDir9am, WindDir3pm, RainToday. Now it gets complicated for machines to understand texts and process them, rather than numbers, since the models are based on mathematical equations and calculations. Therefore, we have to encoded the categorical data with Label Encoder from ski-kit learn. 
      ```python
      from sklearn.preprocessing import LabelEncoder

      le1=LabelEncoder()
      X[:,0] = le1.fit_transform(X[:,0])

      le2 = LabelEncoder()
      X[:,4] = le2.fit_transform(X[:,4])

      le3 = LabelEncoder()
      X[:,6] = le3.fit_transform(X[:,6])

      le4 = LabelEncoder()
      X[:,7] = le4.fit_transform(X[:,7])

      le5 = LabelEncoder()
      X[:,-1] = le5.fit_transform(X[:,-1])

      le6 = LabelEncoder()
      Y[:,-1] = le6.fit_transform(Y[:,-1])
      ```
      
  5. ### Feature Scaling
      
      Our data set contains features with highly varying magnitudes and range. But since, most of the machine learning algorithms use Euclidean distance between two data points in their computations, this is a problem. The features with high magnitudes will weigh in a lot more in the distance calculations than features with low magnitudes. To suppress this eﬀect, we need to bring all features to the same level of magnitudes. This can be achieved by scaling. We used sk-learn’s Standard Scaler to scale all the data points in a certain range.
      ```python
      from sklearn.preprocessing import StandardScaler
      sc = StandardScaler()
      X = sc.fit_transform(X)
      ```
      
* # Modelling

  1. ### Splitting Dataset into Training set and Testing set
      We have split our dataset into training set (80%) and testing set (20%) to train and test rainfall prediction models.
      ```pthon
      from sklearn.model_selection import train_test_split
      X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)
      ```
  
  2. ### Training and Testing 
      We used different classifiers to predict rainfall with our dataset.
      
      1. #### Logistic Regression
          Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables. In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for success/yes) or 0 (stands for failure/no).
          ```python
          from sklearn.linear_model import LogisticRegression  
          classifier_lr = LogisticRegression(random_state=0)  
          classifier_lr.fit(X_train, Y_train)
          ```
          ![image](https://user-images.githubusercontent.com/89296568/140267114-aa73ba55-fbf2-4900-9a2e-6395b3ba3f35.png)

      2. #### Random-Forest
          A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting. 
          ```python
          from sklearn.ensemble import  RandomForestClassifier
          classifier_rfs = RandomForestClassifier(n_estimators=100, random_state=0)
          classifier_rfs.fit(X_train, Y_train)
          ```
          ![image](https://user-images.githubusercontent.com/89296568/140267196-ed2e0b5e-72c9-4c59-a4d9-07a7d2424dd9.png)

          
      3. #### Decision Trees
          Decision tree algorithm falls under the category of supervised learning. They can be used to solve both regression and classification problems. Decision tree uses the tree representation to solve the problem in which each leaf node corresponds to a class label and attributes are represented on the internal node of the tree. We can represent any Boolean function on discrete attributes using the decision tree.
          ```python
          from sklearn.tree import DecisionTreeClassifier  
          classifier_dt = DecisionTreeClassifier(criterion='entropy', random_state=0)  
          classifier_dt.fit(X_train, Y_train)  
          ```
          ![image](https://user-images.githubusercontent.com/89296568/140267319-215acf32-44c2-48fa-a440-8c83f8243cb0.png)

          
      4. #### Light GBM
          LightGBM is a gradient boosting framework based on decision trees to increases the efficiency of the model and reduces memory usage. It uses two novel techniques: Gradient-based One Side Sampling and Exclusive Feature Bundling (EFB) which fulfils the limitations of histogram-based algorithm that is primarily used in all GBDT (Gradient Boosting Decision Tree) frameworks.
          ```python
          from lightgbm import LGBMClassifier
          classifier_lgbm = LGBMClassifier(random_state = 0)
          classifier_lgbm.fit(X_train, Y_train)
          ```
          ![image](https://user-images.githubusercontent.com/89296568/140267380-e369f652-5a0f-4895-b26a-7f5f270bfb26.png)

          
      5. #### Naïve Bayes  
          The naive Bayes classifier assumes all the features are independent to each other. Even if the features depend on each other or upon the existence of the other features. A Gaussian Naive Bayes algorithm is a special type of NB algorithm. It’s specifically used when the features have continuous values. It’s also assumed that all the features are following a gaussian distribution i.e., normal distribution.
          ```python
          from sklearn.naive_bayes import GaussianNB
          classifier_nb = GaussianNB()
          classifier_nb.fit(X_train, Y_train)
          ```
          ![image](https://user-images.githubusercontent.com/89296568/140267497-218bdb5b-7484-4a95-a07a-7d9c0c2cbe74.png)

          
   3. ### Accuracy Comparison
      Finally, we compared accuracies of all the models built and plotted the same using a bar plot.
      
      ![image](https://user-images.githubusercontent.com/89296568/140267530-57f84f11-8cb1-401e-8931-9d69c077c6e4.png)

