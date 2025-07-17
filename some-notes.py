#
# ### Issues with np.log
# Using np.log on csv on the combined array without adjustment causes issues.  These were not fully investigated however it is likely due to fact that csv imports notoriously bring in values with quotes and spaces and suchlike that don't get recognised as numberical values.  (even though inspection on datatype is showing float64)
#
# we therefore use the .to_numeric dataframe function before passing into numpy
#
# ### Issues with to_numeric
# .to_numeric function works on the first row of dataframe since df.iloc[1,1:} returns a pandas series.
# .to_numeric function doesnt work on the whole dataframe however since df.iloc[:,1:] returns another dataframe and not a series
#
#
# ### .stack() and .unstack() solutions
# The DataFrame.stack() function pivots the columns of a DataFrame into the row index, producing a _Series with a MultiIndex_. 
# It reshapes wide-form data into long-form data.
# Wide-form data is where each row is a single observation.
# Long-form data is where each row is a measurement.
# In this use case an observation represents the spot yields at multiple terms for a single date
# and a measurement is a single spot rate at a specific term at a specific date (measurements are more granular than observations)  
#
# .stack() turns columns into rows
# it also converts the datatype from dataframe to series  (which is necessary for our .to_numeric function to work) 
#

