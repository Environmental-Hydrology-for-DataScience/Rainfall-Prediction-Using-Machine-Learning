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
!pip install matplotlib`
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

