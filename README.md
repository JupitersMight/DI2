# DI2 (Distribution Discretizer) Instructions

## Data normalization:

Normalizer.**min_max(pandas.Series)** - Normalizes data by min_max normalization https://en.wikipedia.org/wiki/Feature_scaling#Rescaling

Normalizer.**z_score(pandas.Series)** - Normalizes data z-score normalization https://en.wikipedia.org/wiki/Feature_scaling#Standardization

Normalizer.**mean(pandas.Series)** - Normalizes data by mean normalization https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization

Returns an array of the data after normalization procedure.

## Goodness of fit test

Discretizer.**chi_squared_goodness_of_fit(array, string, integer)** - Applies Person's chi-squared goodness of fit test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

Discretizer.**kolmogorov_goodness_of_fit(array, string, boolean)** - Applies the Kolmogorov–Smirnov goodness of fit test https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

The **chi_squared_goodness_of_fit(array, string, integer)** receives the data (missings must be removed) in form an array, a string with the name of a continuous distribution, from https://docs.scipy.org/doc/scipy/reference/stats.html, and an integer with the number of bins for the test to consider.
Returns a list. [0] : chi-squared statistic : float, [1] : data : array.float

The **kolmogorov_goodness_of_fit(array, string, boolean)** receives the data (missings must be removed) in form an array, a string with the name of a continuous distribution, from https://docs.scipy.org/doc/scipy/reference/stats.html, and a boolean indicating if outliers should be removed (if true the data returned has outliers removed)
Returns a list. [0] : Kolmogorov-Smirnov statistic : float, [1] : data : array.float

## Discretizer

Discretizer.**distribution_discretizer(pandas.Dataframe, integer, optional:string, optional:float, optional:boolean, optional:string, optional:array)** - Discretizes data according to the best fitting distribution.

The **distribution_discretizer(pandas.Dataframe, integer, string, float, boolean, string, array)** receives the data (**pandas.Dataframe**), an **integer** representing the number of categories for discretization, an **string** with the name of the main statistical hypothesis test to apply (options available: **"chi2"**, **"ks"**), a **float** between 0 and 0.49 which indicates the width range to consider a value as being a border value, a **boolean** indicating if outliers should be removed, a **string**  indicating the normalization method to be used (options available: **"min_max"**,**"mean"**, **z_score**), an **array** of continuous distributions, from https://docs.scipy.org/doc/scipy/reference/stats.html, to be considered by the discretizer.

As an illustrative example we use the dataset available at the UCI machine learning repository https://archive.ics.uci.edu/ml/datasets/Breast+Tissue.

---> DI2 was developed by L. Alexandre (leonardoalexandre@tecnico.ulisboa.pt), R.S. Costa (rs.costa@fct.unl.pt) and R. Henriques <---
