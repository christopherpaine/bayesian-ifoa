---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region editable=true slideshow={"slide_type": ""} -->
## transforming dataset via principle component analysis 
To create a parsimonious model for demonstrating the Bayesian technique, we will perform PCA on the raw dataset. This helps reduce noise and extract the most significant patterns of variation. Additionally, it requires fewer parameters than more complex affine term structure models.
<!-- #endregion -->

# Steps
- calculate log of change (don't use df.pcnt_  whatever it is)
- de mean the dataset
- calculate covariance matrix, eigenvalues and eigenvectors
- derive a calibration dataset
- attempt to fit normal or student-t distribution  (student t better for heavier tails)
  - Q Q plot or historgrams  
  - q q plot great for seeing if normally distributed
- ?? BAYESIAN INFERENCE PARTS OF THE PROCESS ??
  - any hyperparameters
- ?? DERIVING STRESSES, COMPARING CLASSICAL VS BAYESIAN APPROACH ??



## Deriving Stresses
$Y_t = \log {\frac{X_t}{X_{t-1}}}$ <br><br>
simulate $Y_t$<br><br>
$ \exp{Y_t} = \frac{X_t}{X_{t-1}} $<br><br>
$X_{t-1}e^{Y_t}  = X_t $<br><br>
$X_t-1$ is current value of the curve and $X_t$ value one year from now


### the steps
- draw realisation of PC from probablistic model
- (if using correlations) rescale using s.d. for each yield maturity
- add back the mean


# 


# Imports

```python
from IPython.display import display, Markdown
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib as plt
```

# Getting Raw Data into Dataframes



We are initially putting the data into dataframes as these enable mixed data types (and we have here a mixture of dates and numerical values)


The bank of england provides two spreadsheets with historic spot yields at https://www.bankofengland.co.uk/statistics/yield-curves/
we import each of these into a dataframes (df1 and df2) and join to make a single dataframe (df)


## Import Libraries


## Load in 1st spreadsheet

```python
df1 = pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=5,header=None)
```

### Headers

```python
col_names=pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=3,nrows=1,header=None)
col_names[0]="Date"
df1.columns = col_names.iloc[0] 

```

### Checksum


#### spreadsheet


A manual highlight of cells in the spreadsheet


![{70A61EF4-CF58-400B-97AF-A62781EC304E}.png](attachment:7fac9a66-b78d-4dd9-9282-c8f578823e43.png)


shows the total value is 191503.17


#### numpy array

```python
df1.shape
```

The values run from row 0, column 1 to row 551, column 50
We put these values into a numpy array and calculate a sum that ignores nil values.  Note np array references are not inlcusive of value after the colon so we add 1 i.e. 0:552,1:52  and NOT 0:551,1:51.

```python
print("The sum of values is from array is "+str(np.nansum(df1.iloc[0:552,1:51].to_numpy()))+" . ")
```

and we see this is the same as sum of values from the spreadsheet.


## Load in 2nd Spreadsheet

```python
#load in second spreadsheet to df2
df2 = pd.read_excel("GLC Nominal month end data_2016 to present.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=5,header=None)
```

### Headers

```python
col_names2=pd.read_excel("GLC Nominal month end data_2016 to present.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=3,nrows=1,header=None)
col_names2[0]="Date"
df2.columns = col_names2.iloc[0] 

```

### Checksum


#### spreadsheet


A manual highlight of cells in the spreadsheet


![{3CA755D4-472C-4E8E-A15F-F9073D665E13}.png](attachment:f9086536-dcab-4f5b-97b7-85fa83be845a.png)


shows the total value is 17845.00


#### numpy array

```python
df2.shape
```

The values run from row 0, column 1 to row 108, column 81
We put these values into a numpy array and calculate a sum that ignores nil values.  Note np array references are not inlcusive of value after the colon so we add 1 i.e. 0:109,1:82

```python
print("The sum of values is from array is "+str(np.nansum(df2.iloc[0:109,1:82].to_numpy()))+" . ")
```

and we see this is the same as sum of values from the spreadsheet.


## Create Combined DataFrame


### Problem of more columns


### Joining 2 datasets

```python
#join the two dataframes to create df
df = pd.concat([df1, df2], ignore_index=True)
print("The length of combined dataframe is "+str(len(df))+" rows")
```

### Check Sum of Values

```python
df.shape
```

```python
print("The sum of values is from combined dataframe is "+str(np.nansum(df.iloc[0:661,1:108].to_numpy()))+" . ")
```

```python
print("The sum of values is from dataframe 1 is "+str(np.nansum(df1.iloc[0:552,1:51].to_numpy()))+" . ")
print("The sum of values is from dataframe 2 is "+str(np.nansum(df2.iloc[0:109,1:82].to_numpy()))+" . ")
print("making a total of "+str(np.nansum(df1.iloc[0:552,1:51].to_numpy())+np.nansum(df2.iloc[0:109,1:82].to_numpy())))
```

### Check Size of Combined DataFrame

```python
#producing some sense checks
display(Markdown("**Checking Dataframe 1 -  1970 to 2015**")) 
print("the first dates is "+ str(df.iloc[0,0].strftime('%Y-%m-%d'))+" and the last is " +str(df.iloc[551,0].strftime('%Y-%m-%d') ))
print("one would therefore expect 12 x 46yrs = 552 entries")
print("and indeed we see the number of rows in df is "+str(len(df1)))
```

```python
display(Markdown("**Checking Dataframe 2 -  2015 to present**")) 
print("the first dates is "+ str(df.iloc[552,0].strftime('%Y-%m-%d'))+" and the last is " +str(df.iloc[659,0].strftime('%Y-%m-%d') ))
print("one would therefore expect 12 x 9yrs = 108 entries")
print("and indeed we see the number of rows in df is "+str(len(df2)))
```

# Choosing Terms to Model


An inspection shows that there are a number of terms for which there is not a continuous set of data points:

```python
nan_summary = df.iloc[0:661,1:108].isna().sum()
ax = nan_summary.plot.bar(edgecolor='black', xlabel="Terms", ylabel="Count of NaN")
ticks = ax.get_xticks();
ax.set_xticks(ticks[::3]);  # Show every 5th label
```

we see that between terms 2 and 15 there are no missing values so we will choose to model this range for our analysis


## NumPy Array with Terms of Interest

```python
df.iloc[0:661,4:31].head()
```

```python
np_ToI = df.iloc[0:661,4:31].to_numpy()
```

# Data Adjustments


Our aim is to model the log differences in spot yields.  However there is a brief period around March 2020 where short term rates dipped below zero. 


![{A2EBC333-AC3D-4237-ABB8-5EAD43D2DB4C}.png](attachment:fcf0d0f0-3b22-4a73-a8e3-2cc850249fcf.png)


These are problematic for the calculation of logs.  For ease of analysis we remove these values and interpolate between the positive values in the month before and after the start of the -ve period


## Removing Negatives

```python
np_ToI_no_negs = np_ToI.copy()
np_ToI_no_negs[np_ToI_no_negs <= 0] = np.nan
```

![{54ED2064-9D24-4AA5-A4D3-E62F44054BFA}.png](attachment:fffdc0b1-41bd-4bd3-9e85-7316665e325f.png)


## Interpolating Gaps

```python
# Convert to a DataFrame for interpolation
df_ToI_no_negs = pd.DataFrame(np_ToI_no_negs)

# Interpolate down the columns
df_ToI_no_negs_interpolated = df_ToI_no_negs.interpolate(method='linear', axis=0)

# Convert back to NumPy
np_ToI_no_negs_interpolated = df_ToI_no_negs_interpolated.to_numpy()
```

## Log Yields


we calculate the natural log of the spot yields

```python
np_ToI_logged = np.log(np_ToI_no_negs_interpolated)
```

![{9BBABC4B-2728-4BC1-9630-0BFBFD6DCA5D}.png](attachment:bbae329c-8bf9-47b7-a0b5-ace76a01d3e2.png)


## Calculating Yield Differences


### Spot Yield Differences


For each term of interest we calculate value of each natural log spot rate less the value of the natural log spot rate 1 year prior

```python
lag = 12
diffs = np_ToI[lag:] - np_ToI[:-lag]

```

### sense check


We are interested in terms 2 to 15.  This includes half years so we have 27 columns.
There are 660 rows of data.  Since differences use a lag of 12 we will have 12 less rows  648.

```python
print(np_ToI[0:648].shape)
print(np_ToI[12:661].shape)
```

The total value of differences should equate to the sum of rows 12:661 less sum of rows 0:648

```python
np.sum(np_ToI[12:661])-np.sum(np_ToI[0:648])
```

```python
np.sum(diffs)
```

### spot check


 <span style="color:red">we work through a couple of examples from raw data to the end to make sure same in output</span>


## De Mean


### Calc Mean for Each Column and Deduct

```python
# Compute the mean of each column
column_means = diffs.mean(axis=0)
print(column_means)
# Subtract the column means from the original array
demeaned_A = diffs - column_means
```

### Sense Checks


#### Check the means

```python
diffs.shape
```

The sum of means * number of rows should aggregate to the sum of all entrie

```python
(column_means *648).sum()
```

```python
diffs.sum()
```

#### Check the de-meaned dataset


The sum of values in a demeaned dataset should equal zero.  Furthermore, the sum of values in the original dataset less the mean * number of rows should also equal zero.

```python
round(demeaned_A.sum(),7)
```

```python
print(diffs.sum() - (column_means *648).sum())
```

### spot check


 <span style="color:red">we work through a couple of examples from raw data to the end to make sure same in output</span>


#  <span style="color:red">Covariance or Correlation</span>

```python
type(demeaned_A)
```

```python
covariance_matrix = np.cov(demeaned_A)
correlation_matrix = np.corrcoef(demeaned_A)
```

# Eigenvectors and Eigenvalues

```python
eigenvalues_from_covmatrix, eigenvectors_from_covmatrix = np.linalg.eig(covariance_matrix)
```

```python
eigenvalues_from_correlmatrix, eigenvectors_from_correlmatrix = np.linalg.eig(correlation_matrix)
```

# 


# Incorporating Bayesian Framework


PCA reveals latent factors (


Having decided to model interest principle components, which economic outlooks correspond to these components:


|Principle Component   |Relevant Insights   |
|---|---|
|PC1|level of interest rates  -  expected prolonged rates or gradual hiking against prolonged inflation |
|PC2|slope  -  short term vs long term expectations   |
|PC3|curvature  -   short term vs long term expectations   |




coud the bayesian rules enforce arbitrage freeness ??


- Economic theory imposes contraints of the first moments (see https://www.nber.org/system/files/working_papers/w24618/w24618.pdf)


# Some links
https://www.thegoldensource.com/pca-and-the-term-structure/#:~:text=The%20purpose%20of%20PCA%20is,14%20orthogonal%20lines%20using%20eigenvectors.

```python

```
