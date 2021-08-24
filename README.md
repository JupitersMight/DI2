# DI2 (Distribution Discretizer)

## Import

import DI2

## Discretizer

**distribution_discretizer(dataset, number_of_bins, statistical_test, cutoff_margin, kolmogorov_opt, normalizer, distributions, single_column_discretization)**

**distribution_discretizer(pandas.Dataframe, integer, optional:string, optional:float, optional:boolean, optional:string, optional:array)** - Discretizes data according to the best fitting distribution.

The **distribution_discretizer(pandas.Dataframe, integer, string, float, boolean, string, array)** receives the data (**pandas.Dataframe**), an **integer** representing the number of categories for discretization, an **string** with the name of the main statistical hypothesis test to apply (options available: **"chi2"**, **"ks"**), a **float** between 0 and 0.49 which indicates the width range to consider a value as being a border value, a **boolean** indicating if outliers should be removed, a **string**  indicating the normalization method to be used (options available: **"min_max"**,**"mean"**, **z_score**), an **array** of continuous distributions, from https://docs.scipy.org/doc/scipy/reference/stats.html, to be considered by the discretizer.


## Data normalization:

Normalizes data by min_max normalization https://en.wikipedia.org/wiki/Feature_scaling#Rescaling

Normalizes data z-score normalization https://en.wikipedia.org/wiki/Feature_scaling#Standardization

Normalizes data by mean normalization https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization


## Goodness of fit test

Person's chi-squared goodness of fit test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

Kolmogorov–Smirnov goodness of fit test https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test


As an illustrative example we use the dataset available at the UCI machine learning repository https://archive.ics.uci.edu/ml/datasets/Breast+Tissue.

---> DI2 was developed by L. Alexandre (leonardoalexandre@tecnico.ulisboa.pt), R.S. Costa (rs.costa@fct.unl.pt) and R. Henriques <---
