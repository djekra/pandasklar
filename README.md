# pandasklar
Toolbox / Ecosystem for data science. Easier handling of pandas, especially when thinking in SQL.

Focused on working with complex, ambiguous, erroneous, two-dimensional DataFrames containing one- or two-dimensional objects.
Focused on convenience when working with jupyter notebooks, not speed (exception: `fast_startswith` and `fast_endswith`).

Convenience means: 
* use more high-level functions
* functions try to cope even with sloppy data and try to avoid error messages when running cells again
* basic functions easier to remember

Comes in the form of helper functions, i.e. without changes to pandas, just on top of it.
Full dokumentation see `jupyter` directory.


## Install
`pip install pandasklar`


## Create Random Data for Testing
* `random_series`: Returns a series of random data of several types, including names, random walks with perlin-noise and errorprone series to test your functions.
* `decorate`: Decorates a series with specials (e.g. NaNs)
* `people` and `random_numbers`: Random data for testing.


## Review and visualize
* `plot`: Plot with seaborn without configuration
* `check_mask`: Count rows filtered by a binary mask. Raises an error, if the number is unexpected.
* `sample`: Returns some sample rows: beginning + end + random rows in the middle, prefering rows without NaNs
* `search_str`: Searches all str columns of a dataframe. Useful for development and debugging.
* `grid`: Visualize with dtale


## Analyse Datatypes
* `analyse_datatypes`: Returns info about the datatypes and the mem_usage of the columns of a DataFrame 
* `analyse_values`: Returns statistical data for a DataFrame, a Series or an Index 
* `analyse_cols`: Describes the datatypes and the content of a DataFrame. Merged info from analyse_datatypes and analyse_values
* `change_datatype`: Converts the datatypes of a DataFrame or a Series. Automatically, if you want.
* `copy_datatype`: Copies the dtypes from one dataframe to another, matching the column names.


## Analyse Frequencies
* `analyse_freqs`: Frequency analysis that includes a subordinate frequency analysis. Provides e.g. the most important examples per case. Splits strings and lists.


## Analyse uniqueness, discrepancies und redundancy
* `analyse_groups`: Analyses a DataFrame for uniqueness and redundancy.
* `same_but_different`: Returns the rows of a DataFrame that are the same on the one hand and different on the other: They are the same in the fields named in same. And they differ in the field named in different. This is useful for analysing whether fields correlate 100% with each other or are independent.


## Compare Series and DataFrames
* `compare_series`: Compares the content of two Series. Returns several indicators of equality.
* `compare_dataframes`: Compares the content of two DataFrames column by column.<br>
   Returns several indicators of equality.
* `check_equal`: Compares the content of two DataFrames column by column.
* `compare_col_dtype`: Returns the column names of two DataFrames whose dtype differs
* `get_different_rows`: Returns the rows of two DataFrames that differ


## Manage columns
* `drop_cols`: Drops a column or a list of columns. Does not throw an error if the column does not exist.
* `move_cols`: Reorders the columns of a DataFrame. The specified columns are moved to a numerical position or behind a named column.
* `rename_col`: Renames a column of a DataFrame. If you try to rename a column again, no error is thrown (better for the workflow in jupyter notebooks).
* `col_names`: Selects column names based on analyse_cols. Useful to apply a method to specific columns of a DataFrame.


## Manage rows
* `drop_multiindex`: Converts any MultiIndex to normal columns and resets the index. Works with MultiIndex in Series or DataFrames, in rows and in columns.
* `reset_index`: Creates a new, unnamed index. If keep_as is given, the old index is preserved as a row with this name. Otherwise the old index is dropped.
* `rename_index`: Renames the index.
* `drop_rows`: Drops rows identified by a binary mask, verbose if wanted.
* `move_rows`: Moves rows identified by a binary mask from one dataframe to another (e.g. into a trash).<br>
   The target dataframe gets an additional message column by standard (to identify why the rows were moved). Verbose if wanted. 
* `add_rows`: Like concat, with additional features only_new and verbose.


## Let DataFrames Interact
* `isin`: isin over several columns. Returns a mask for df1: The rows of df1 that match the ones in df2 in the specified columns.
* `update_col`:     Transfers one column of data from one dataframe to another dataframe.<br>
   Unlike a simple merge, the index and the dtypes are retained. Handles dups and conditions. Verbose if wanted.


## Create DataFrames Easily
* `dataframe`: Converts multidimensional objects into dataframes. Dictionaries and Tuples are interpreted column-wise, Lists and Counters by rows.


## Save and load data
* `dump_pickle`: Convenient function to save a DataFrame to a pickle file. Optional optimisation of datatypes. Verbose if wanted.
* `load_pickle`: Convenient function to load a DataFrame from pickle file. Optional optimisation of datatypes. Verbose if wanted.
* `dump_excel`: Writes a dataframe into an xlsx file for Excel or Calc.<br>
   The tabcol-feature groups by a specific column and creates a tab for every group.
* `load_excel`: Loads a dataframe from an xlsx file (Excel or Calc).<br>
   The tabcol-feature writes all tabs in a column of the result.
   
   
## Work with NaN
* `nnan`: Count NaNs in Series or DataFrames.
* `any_nan`: Are there NaNs? Returns True or False.
* `nan_rows`: Returns the rows of a DataFrame that are NaN in the specified column.


## Scale Numbers
* `scale`: Scales all values of a numeric series to a defined value range.<br>
   Available methods: max_abs, min_max, min_max_robust, rel, mean, median, 
   compare_median, rank and random.


## Cleanup Strings
* `remove_str`: Removes a list of unwanted substrings from a Series of strings.
* `remove_words`: Removes a list of unwanted words from a Series of strings.
* `replace_str`: Replaces substrings from a Series of strings according to a dict.


## Slice Strings Variably
* `slice_string`: Slices a column of strings based on indexes in another columns.


## Search Strings Fast
* `fast_startswith`: Searches string columns for matching beginnings.<br>
   Like pandas str.startswith(), but much faster for large amounts of data, and it returns the matching fragment.
* `fast_endswith`: Searches string columns for matching endings.


## Work with Lists
* `find_in_list`: Searches a column with a list of strings. Returns a binary mask for the rows containing the searchstring in the list. 
* `apply_on_elements`: Applies a function to all elements of a Series of lists.
* `list_to_string`: Converts a Series of lists of strings into a Series of strings.


## Rank Rows
* `rank`: Select the max row per group. Or the min.<br>
   Or mark the rows instead of selecting them. 


## Aggregate Rows
* `group_and_agg`: Groups and aggregates. Provides a user interface similar to that of MS Access.
* `most_freq_elt`: Aggregates a Series to the most frequent scalar element.<br>
   Like Series.mode, but always returns a scalar.
* `top_values`: Aggregates a Series to a list of the most frequent elements.<br>
   Can also return the counts of the most frequent elements.  
* `agg_words`: Aggregates a Series of strings to a long string.<br>
   A space is always placed between the elements, the order is preserved.
* `agg_words_nodup`: Aggregates a Series of strings (e.g. signal words) to a long string.
   Like agg_words, but duplicates are removed.
* `agg_strings_nospace`: Aggregates a Series of strings into one long string.<br>
   Like agg_words, but no separators between the substrings.   
* `agg_to_list`: Aggregates a Series to a list. 
   Normally this can also be done with a simple 'list', 
   but in combination with transform this does not work.
   Then agg_to_list can be used as a substitute.
* `agg_dicts`: Aggregates a Series of dicts to a single dict.<br>
   If a key occurs more than once, the value is overwritten.
* `agg_dicts_2dd`: Aggregates a Series of dicts to a single defaultdict(list).<br>
   I.e. multiple keys are allowed. The values are always lists. 
* `agg_defaultdicts`: Aggregates a Series of defaultdict(list) to a single defaultdict(list). 


## Explode and Implode Dictionaries
* `explode_dict`: Like pandas explode, but for a dict.<br>
  Turns dictionaries into two columns (key, value) and additional rows, if needed.
* `implode_to_dict`: Reversal of explode_dict.<br>
  Groups rows and turns two columns (key, value) into one dict. 
* `cols_to_dict`: Moves columns into a dict or defaultdict. 
