# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Bayesian Principal Component Analysis 

# %%
# basic imports
from IPython.display import display, Markdown, HTML, SVG
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO







# load in first spreadsheet to df1
df1 = pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=5,header=None)
# create an appropriate set of headers
col_names=pd.read_excel("GLC Nominal month end data_1970 to 2015.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=3,nrows=1,header=None)
col_names[0]="Date"
df1.columns = col_names.iloc[0] 
# load in second spreadsheet to df2
df2 = pd.read_excel("GLC Nominal month end data_2016 to present.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=5,header=None)
# create an appropriate set of headers
col_names2=pd.read_excel("GLC Nominal month end data_2016 to present.xlsx",sheet_name="4. spot curve",engine="openpyxl",skiprows=3,nrows=1,header=None)
col_names2[0]="Date"
df2.columns = col_names2.iloc[0]

# %%
#join the two dataframes to create df
df = pd.concat([df1, df2], ignore_index=True)

# %%

fred = "the first date is "+ str(df.iloc[0,0].strftime('%Y-%m-%d'))+" and the last is " +str(df.iloc[551,0].strftime('%Y-%m-%d') )
df2_dates = "the first dates is "+ str(df.iloc[552,0].strftime('%Y-%m-%d'))+" and the last is " +str(df.iloc[659,0].strftime('%Y-%m-%d') )
df1_length = str(len(df1))
df2_length = str(len(df2))
df1_sum = str(df1.iloc[:, 1:].sum().sum())
df2_sum = str(df2.iloc[:, 1:].sum().sum())
df_sum = str(df.iloc[:, 1:].sum().sum())
combined_total = 191503.172322029 + 17844.9993308767
df_length = str(len(df)) 
display(HTML(f"""
<div style="display: flex; padding: 5px;">
  <div style="flex: 1; padding: 5px;">            <h2>Creating One Combined DataFrame</h2>
            <p>We have 2 spreadsheets of spot yields from the Bank of England website that we will load into dataframes</p>
            <a href="./GLC Nominal month end data_1970 to 2015.xlsx" download>Download GLC Nominal month end data_1970 to 2015.xlsx</a><br>
            <a href="./GLC Nominal month end data_2016 to present.xlsx" download>Download GLC Nominal Month End Data (2016 to Present)</a>
</div>
  <div style="flex: 1; padding: 5px;">
             <h2>A summary of the process</h2>
                <div style="list-style-type: square;">
                  <ul>
                    <li>Load BoE data into one dataframe</li>
                    <li>Truncate the data so that continous block of data available for calibration</li>
                    <li>Interpolate the data so that continous block of data available for calibration</li>
                    <li>Remove negative values and interpolate between remaingin values<li>
                    <li>Take Logarithms</li>
                    <li>Difference the data</li>
                    <li>De-mean the data</li>
                    <li>Calculate co-variance matrix</li>
                    <li></li>
                  </ul>
                </div>
</div>
</div>



<h3>Basic Reasonableness Tests</h3>
<p>We perform a couple of reasonableness checks to ensure the spreadsheet data has loaded correctly into the combined dataframe</p>
<div style="display: flex; gap: 20px;">
  <div style="flex: 1;border: 1px solid #999;padding: 10px;">
    <h4><u>A Check on the Number of Rows</u></h4>
    <div style="display: flex; gap: 5px;">
      <div style="flex: 1;">
        <h5>Dataframe 1 - 1970 to 2015</h5>
        <p>{fred}<br>
        one would therefore expect 12 x 46yrs = 552 entries<br>
        and indeed we see the number of rows in df is {df1_length}</p>
      </div>
      <div style="flex: 1;">
        <h5>Dataframe 2 - 2015 to present</h5>
        <p>{df2_dates}<br>
        one would therefore expect 12 x 9yrs = 108 entries<br>
        and indeed we see the number of rows in df is {df2_length}</p>
     </div>
    </div>
    <h5>Combined DataFrame</h5>
<p>The length of combined dataframe is {df_length} rows"<br>
        whereas the two separate dataframes come to 552 + 108</p>
  </div>
  <div style="flex: 1;border: 1px solid #999;padding: 10px;">
    <h4><u>A Check on Sum of Values</u></h4>
    <div style="display: flex; gap: 5px;">
      <div style="flex: 1;">
    <h5>Dataframe 1 - 1970 to 2015</h5>
    <p>manual inspection of the sum of all values in first spreadsheet is 191503.172322029<br>the sum of 1st dataframe is also {df1_sum}</p> 
      </div>
      <div style="flex: 1;">
    <h5>Dataframe 2 - 2015 to present</h5>
    <p>manual inspection of the sum of all values in second spreadsheet is 17844.9993308767<br>the sum of 1st dataframe is also {df2_sum}</p> 
      </div>
      </div>
    <h5>Combined DataFrame</h5>
        <p>the sum of combined dataframe is {df_sum}<br>
and the sum of the manually observed 191503.172322029 + 17844.9993308767 = {combined_total}</p>
  </div>
</div>
<hr>
"""))




# %%
# %%capture
# bar chart 
combined_df_isna_count = df.iloc[0:,1:].isna().sum().to_frame()
combined_df_isna_count_html = df.isna().sum().to_frame().to_html()


fig, ax = plt.subplots(1, 1);
combined_df_isna_count.plot(kind='bar', y=0, ax=ax);  # attach plot to your Figure
ax.set_title("Null Counts by Term");
ax.set_ylabel('No of Nulls');
ax.legend().remove();
xticks = ax.get_xticks();
ax.set_xticks(xticks[::5]);  # Keep every 5th tick

# Save figure to SVG string
buf = StringIO()
fig.savefig(buf, format='svg')
svg_str = buf.getvalue()
buf.close()
with open("plot.svg", "w") as f:
    f.write(svg_str)


# %%
# %%capture
# first and last non null
# Get the index of the first non-null value for each column
# For each column in the DataFrame, find the index of the first non-null value, and return a Series mapping column names to those index labels.
first_non_null = df.apply(lambda col: col.first_valid_index())

# Get the index of the last non-null value for each column
last_non_null = df.apply(lambda col: col.last_valid_index())

# determine key values for the y-axis
# we want to create a series of key dates
unique_y = np.concatenate((first_non_null.unique(),last_non_null.unique()))
y_dates = df.iloc[unique_y,0]
y_dates = [dt.strftime('%Y-%m-%d') for dt in y_dates]


# DataFrame for line chart
lineg = pd.DataFrame(
    [first_non_null.values, last_non_null.values],
    columns=first_non_null.index  # or .keys()
)
x = lineg.columns[1:].astype(float).to_numpy()
y1 = lineg.iloc[0,1:].astype(float)
y2 = lineg.iloc[1,1:].astype(float)
fig2, ax2 = plt.subplots(1,1)
ax2.plot(x, y1,label='first non null')
ax2.plot(x, y2,label='last non null')
ax2.legend()
ax2.set_yticks(ticks=unique_y,labels=y_dates)
fig2.subplots_adjust(left=0.25)#so we can see the whole date label



# Save figure to SVG string
buf = StringIO()
fig2.savefig(buf, format='svg')
svg_str = buf.getvalue()
buf.close()
with open("plot2.svg", "w") as f:
    f.write(svg_str)


# Tabular presentation of data boundaries by term
step1 = pd.DataFrame(first_non_null[1:].index)
step1.columns = ['Terms']
f
first_non_null[1:]
dir(step1)
step2 =step1.copy()
step2['earliest-row-loc'] = first_non_null[1:].values
step3 = step2.copy()
step3["earliest-date"] = step3["earliest-row-loc"].apply(lambda x: df.iloc[x, 0])
step4 = step3.copy()
step4['last-row-loc'] = last_non_null[1:].values
step5 = step4.copy()
step5["last-date"] = step5["last-row-loc"].apply(lambda x: df.iloc[x, 0])
step6 = step5.copy()
step6["date_pair"] = list(zip(step5["earliest-date"], step5["last-date"]))
step6["group_id"] = (step6["date_pair"] != step6["date_pair"].shift()).cumsum()
step7 = (
    step6.groupby("group_id")
    .agg(
        start_term=("Terms", "min"),
        end_term=("Terms", "max"),
        earliest_date=("earliest-date", "first"),
        last_date=("last-date", "first"),
    )
    .reset_index(drop=True)
)
step8 = step7.to_html(index=False)



# Expected Data Points vs Actual Data Points
# ------------------------------------------
EvAstep1 = step7.copy()
EvAstep1['earliest_date'] = pd.to_datetime(EvAstep1['earliest_date'])
EvAstep1['last_date'] = pd.to_datetime(EvAstep1['last_date'])
EvAstep2 = EvAstep1.copy()
EvAstep2['num_months'] = ((EvAstep2['last_date'].dt.year - EvAstep2['earliest_date'].dt.year) * 12 +
                          (EvAstep2['last_date'].dt.month - EvAstep2['earliest_date'].dt.month) + 1)
EvAstep3 = pd.DataFrame(df.count())
EvAstep3 = EvAstep3.reset_index(names='date')
EvAstep3.columns = ['term', 'value']
EvAstep3 = EvAstep3.iloc[1:].reset_index(drop=True)
EvAstep3.columns = ['term', 'actual no. data-points']

def lookup_function(row,reference_df):
    matched_row = reference_df[(reference_df['start_term']<=row['term'])&(reference_df['end_term']>=row['term'])]
    return matched_row['num_months'].values[0] if not matched_row.empty else None

EvAstep4 = EvAstep3.copy()
EvAstep4['expected no. data-points'] = EvAstep4.apply(lambda row: lookup_function(row,EvAstep2), axis=1)
EvAstep4['missing data points']=EvAstep4.apply(lambda row: row.iloc[2]-row.iloc[1],axis=1)
EvAstep5 = EvAstep4.copy()
EvAstep5 = EvAstep5[EvAstep5.iloc[:,3]>0]
EvAstep6 = EvAstep5.to_html(index=False)

#truncation and interpolation of the dataset
#-------------------------------------------
date_col = df.iloc[:,0]
numeric_cols_interpolated = df.iloc[:,1:].interpolate()
df_interpolated = pd.concat([date_col,numeric_cols_interpolated],axis=1)
#truncation
df_truncated = df_interpolated.drop(df_interpolated.columns[41:],axis=1)
df_truncated = df_truncated.drop(df_truncated.columns[1],axis=1)




# Some notes on variables we have set
# -----------------------------------
# first_non_null and last_non_null are each panda series and each return a row index 
# unique_y is a numpy array that gives us unique row indexes at which first or last observations occur 
# y_dates is a list that gives us unique dates at which first or last observations occur 
# y1 and y2 are row indexes for all terms of first and last observations respectively


# %%
display(HTML(f"""
<h2>Truncation & Interpolation of the Dataset</h2>

<div style="display: flex; padding: 5px;">
  <div style="flex: 1;">
    <!-- Left column content -->
    <p>Principal component analysis requires same number of datapoints for each term so as to produce a rectangular matrix from which covariances can be calculated.</p>
    <p>The dataset of spot yields contains gaps insofar that the whole set of observation dates is not consistently available for all terms.  We want to choose a range of observation dates and terms that reduces the need to fill in gaps in the dataset.</p>
    <p>We have spot yield data for terms 0.5 up to 40.  The first step to identify a calibration dataset is to identify the first and last data point for each term.  This gives us an initial idea of the size of the dataset available.</p>
    <p>We make a judgement call about which terms to retain (and observation dates) to retain.  If there are gaps in the data we use linear interpolation to fill them.</p>
</div>
  <div style="flex: 1; background-color: #ddd;border: 1px solid #999;padding: 10px;">
    <!-- Right column content -->


<h3>Matplotlib figures, subplots, axes</h3>
<ul>
    <li>Figure	The whole canvas or image </li>
    <li>Axes	One chart (with x/y axes, labels, data) </li>
    <li>Subplot	One chart within a grid layout (i.e., an Axes) </li>
    <li>Grid of subplots	Arrangement of multiple Axes in a Figure</li>
</ul>
  </div>
</div>




<div style="display: flex; padding: 5px;">
  <div style="flex: 1;">
    <h3>Data Boundaries by Term</h3>
             <u> <h5>visual</h5> </u>
             <p>The maiximum range of observation dates for each term is found by the earliest and latest non NaN entry.  We see that for beyond term 25 data is only available from  31st January 2016 and that for earlier terms available from 31st January 1970 (with an exception for term 0.5).</p>
    <img src="plot2.svg" alt="My chart">
    <u> <h5>tabular</h5> </u>
    {step8}
    <h3>Interpolation</h3>
    <p>Summary statistics on interpolated/truncated dataset</p>
    <ul>
        <li>term</li>
        <li>actual data points</li>
    </ul>
    <p>Rows untouched by interpolation should have same total as before.  totals for those with interpolation should could be checked for reasonableness. </p>
    <h3>Decisions</h3>
    <ul>
        <li>Data for terms greater than 25 isn't available before 2016.  We will therefore not model beyond term 25 in order to facilitate sufficient history of data-points. </li>
        <li>we ignore term 0.5 and start at term 1 due to missing datapoints for term 0.5</li>
        <li>we ignore terms beyond 20 since the proportion of missing datapoints is too great.</li> 
        <li>we replace -ve values with NaN and then interpolate</li>
    </ul>

   </div>
  <div style="flex: 1;">
    <h3>Null Counts</h3>
    <h4>Histogram</h4>
    <p>An initial inspection of the data shows signficantly more nulls for greater terms.  Beyond term 25 we see the number levels off and we later discover this is because data for term 25 onwards doesn't begin until 2016 meaning there is a significant block of NaN values from 1970 to 2016 for these terms.</p>
    <img src="plot.svg" alt="My chart">
    <h4>Tabulated</h4>
        <p>We identify non contiguous blocks of data by determining the expected number of data points, based on first and last data point, and comparing with actual number of data points. </p>
        <p>These are the columns which will be interpolated.</p>
    {EvAstep6}
  </div>
</div

"""             ))


# %%
identifying_negatives = df_truncated.iloc[:,1:][(df_truncated.iloc[:,1:]<0).any(axis=1)]
df_truncated_no_negs = df_truncated.iloc[:,1:].where(df_truncated.iloc[:,1:]>=0,np.nan)
df_truncated_no_negs_interpolated = df_truncated_no_negs.interpolate()
display(HTML(rf"""
<hr>
<h2>Removing Negatives </h2> 
             <p>
             Logarithms are only defined for positive arguments.  We therefore need to consider the small number of -ve values observable in the dataset:
             </p>
             {identifying_negatives.to_html()}
             For ease of analysis we set these values to NaN.
             {df_truncated_no_negs.loc[identifying_negatives.index].to_html()}
             We now populate this values with interpolated values moving down the columns (terms) 
             {df_truncated_no_negs_interpolated.loc[identifying_negatives.index].to_html()}
<span style='color:red;font-size:10px;'>checks we can perform on the interpolated values .....</span>
            <hr>
"""             ))


# %% [markdown]
# ## Taking Logarithmns
#


# %%
# %%capture
df_logged = df_truncated_no_negs_interpolated.astype(float).apply(np.log) 
df_log_sum = df_logged.sum().sum()

logcheck=pd.DataFrame({"the product of each row":df_truncated_no_negs_interpolated.product(axis=1),"the log of each row product":df_truncated_no_negs_interpolated.product(axis=1).apply(np.log),"the sum of log of row products":df_truncated_no_negs_interpolated.product(axis=1).apply(np.log).sum()})
sum_of_log_row_prdocuts=df_truncated_no_negs_interpolated.product(axis=1).apply(np.log).sum()
logcheck=logcheck.iloc[0:5,:].to_html(index=False)





# %%
html = fr"""
<div style="display: flex; gap: 20px;">
  <div style="flex: 1;">
    <h3>Purpose</h3>
      <p>We want to calculate the natural log of spot yield returs.  <span style='color:>method</span></p>
        <p>\[
                
        \]</p>
      <h3>Applying Natural Log to the Whole DataFrame</h3>
      <p>The sum of all the individual 'logged' values is: </p>
      {df_log_sum}
  </div>
  <div style="flex: 1; background-color: #ddd;border: 1px solid #999;padding: 10px;">
    <h3>.apply() function</h3>
    <p>This is the second column. Same flexibility as the first.</p>
    <h3>further complications with dtype:object</h3>
      <p>sometimes pandas is treating values as generic python objects not efficient numeric types even if they look like floats</p>
      <p>it seems to happen when slicing rows.</p>
      <p>a fix is to use .astype(float) before applying functions like np.log</p>
   </div>
</div>
<div style="display: flex; gap: 20px;">
  <div style="flex: 1;">
    <h3> Checking the Log Calculation</h3>
    given that:
        <p>\[
        \sum_i \log(x_i) = \log\left( \prod_i x_i \right)
        \]</p>
    <p>we can perform a check on the log calculation. however the product approach doesn't work since there are so many values we get overflow for the product side of the equation we can instead chunk up the calculation to make it more manageable we therefore calculate the product for each row then take the log the sum the log of products for each row</p>
    
    

   </div>
  <div style="flex: 1; padding: 10px;">
    <h3>The Product of All Entries</h3>
    {logcheck}
   </div>
</div>
"""

display(HTML(html))

# %%
error_diff = sum_of_log_row_prdocuts - df_logged.sum().sum()

display(HTML(rf"""
<h3>Comparing Calculations</h3>         
<p>The sum of individual 'logged values is:  {str(sum_of_log_row_prdocuts)}.  The sum of the log of row product generates: {str( df_logged.sum().sum())}.  The difference between the two is {str(error_diff)}.</p>
"""             ))

# %%
# we inherit df_logged dataframe and we create df_log_differenced 
df_log_differenced = df_logged.diff()
first_row = df_logged.iloc[1,:].sum()
last_row = df_logged.iloc[-1,:].sum()
df_log_differenced_sum = df_log_differenced.sum().sum()
check_value = first_row - last_row

#display(HTML("<h3>Input DataFrame B</h3>"))
#display(df2.head(5).style.set_caption("DataFrame B"))
#
#display(HTML("<h3>Difference (A - B)</h3>"))
#display((df1 - df2).head(5).style.background_gradient(cmap="RdBu"))

def highlight_locations(x):
    styles = pd.DataFrame('', index=x.index, columns=x.columns)
    styles.iloc[3, 3] = 'background-color: #fff8b0'  # light pastel yellow
    styles.iloc[4, 3] = 'background-color: #fff27a' 
    return styles


def highlight_locations2(x):
    styles = pd.DataFrame('', index=x.index, columns=x.columns)
    styles.iloc[4, 3] = 'background-color: #ffd6d6'  # light pastel yellow
    return styles

display(HTML(rf"""
<hr>
<h2>Differencing Data</h2>

<div style="display: flex; padding: 5px;">
  <div style="flex: 1; padding: 0px;">
  <p>We calculate differences since we are modelling changes in the yield curve:</p> 




    </div>
  <div style="flex: 1; padding: 5px;background-color: #ddd;border: 1px solid #999;">
             <b>
             <p>Checks that can be made to ensure data has been differenced correctly: </p>
             </b>

        <div style="list-style-type: square;">
          <ul>
            <li>spot check a small sample of values</li>
            <li>total of differences = sum of first row - sum of last row</li>
          </ul>
        </div>

  </div>
</div>


<div style="display: flex; padding: 5px;">
  <div style="flex: 1; padding: 5px;">

             <h3>Logged Values</h3>
             {
                 df_logged.iloc[:,:10].head().style.apply(highlight_locations, axis=None).to_html()
                 }

              <h3>Differenced Values</h3>

             {
                 df_log_differenced.iloc[:,:10].head().style.apply(highlight_locations2, axis=None).to_html()
                 }
  


             </div>
  <div style="flex: 1; padding: 5px;">


            <h3>
             Spot Checks
             </h3>
                <p>
                <span style="background-color: #fff27a; padding: 2px 4px;">{df_logged.iloc[3,3]}</span>
                  minus
                  <span style="background-color: #fff8b0; padding: 2px 4px;">{df_logged.iloc[4,3]}</span>
                  equals
                  <span style="background-color: #ffd6d6; padding: 2px 4px;">{ df_logged.iloc[3,3]-df_logged.iloc[4,3] }</span>
                </p>
             <h3>
             Aggregate Checks
             </h3>
             <p>the sum of the differences is:{df_log_differenced_sum}</p>  
                <p>the sum of the first row minus the last row is:{check_value}</p>  

<span style='color:red;font-size:16px;'>narrow down the difference</span>



             </div>
</div>

"""             ))


# %%
df_demeaned = df - df.mean()
display(HTML(rf"""
<hr>
<h2>De-meaning Data</h2>



"""             ))



# %%
covariance_matrix = df_logged.cov()
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

display(HTML(rf"""
<h2>Co-variance Matrix</h2>


"""             ))





## Write to Excel
#from openpyxl import load_workbook
#file = "./spreadsheets/what-went-wrong.xlsx"
#with pd.ExcelWriter(file, mode='w') as writer:
#    combined.to_excel(writer, sheet_name="summary",index=False)
#    df_log_diff.to_excel(writer,sheet_name="df_log_diff",index=False, engine='openpyxl')
#    df_logged.to_excel(writer,sheet_name="df_logged",index=False, engine='openpyxl')
#    df_truncated.to_excel(writer,sheet_name="df_truncated",index=False, engine='openpyxl')
#
## Adjust column widths
#wb = load_workbook(file)
#ws = wb['summary']
#for col in ws.columns:
#    max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in col)
#    ws.column_dimensions[col[0].column_letter].width = max_length + 2
#
#wb.save(file)




