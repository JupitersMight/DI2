"""
DistFitting
=====

Provides
  1. A data Normalizer
  2. A data discretizer
  3. Goodness of fit tests between two continuous data distributions

----------------------------
Documentation is available at github.

"""

import pandas as pd
import scipy.stats as scp
import scipy.stats
import scipy
import numpy as np
import math
import scipy.stats
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings("ignore")

# Chi squared statistic to p-value table
chi_squared_table = {
    0.05: {  # pvalue < 0.05
        1: 3.841,
        2: 5.991,
        3: 7.815,
        4: 9.488,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919,
        10: 18.307,
        11: 19.675,
        12: 21.026,
        13: 22.362,
        14: 23.685,
        15: 24.996,
        16: 26.296,
        17: 27.587,
        18: 28.869,
        19: 30.144,
        20: 31.410,
        21: 32.671,
        22: 33.924,
        23: 35.172,
        24: 36.415,
        25: 37.652
    }
}
# One-Sample Kolmogorov-Smirnov Table
kolmogorov_table = {
    0.05: {  # pvalue < 0.05
        1: 0.975,
        2: 0.842,
        3: 0.708,
        4: 0.624,
        5: 0.563,
        6: 0.519,
        7: 0.483,
        8: 0.454,
        9: 0.430,
        10: 0.409,
        11: 0.391,
        12: 0.375,
        13: 0.361,
        14: 0.349,
        15: 0.338,
        16: 0.327,
        17: 0.318,
        18: 0.309,
        19: 0.301,
        20: 0.294,
        21: 0.287,
        22: 0.281,
        23: 0.275,
        24: 0.269,
        25: 0.264,
        26: 0.259,
        27: 0.254,
        28: 0.250,
        29: 0.246,
        30: 0.242,
        31: 0.238,
        32: 0.234,
        33: 0.231,
        34: 0.227,
        35: 0.224,
        36: 0.221,
        37: 0.218,
        38: 0.215,
        39: 0.213,
        40: 0.210
        # n > 40 = 1.36/raizquadrada(n)
    }
}


def _get_normalized_data(column, normalizer):
    y_std = []
    if normalizer == "min_max":
        y_std = Normalizer.min_max(column.dropna())
    elif normalizer == "mean":
        y_std = Normalizer.mean(column.dropna())
    elif normalizer == "z_score":
        y_std = Normalizer.z_score(column.dropna())
    return y_std


def _get_column_properties(column):
    return {
        "min": column.astype(float).dropna().min(),
        "max": column.astype(float).dropna().max(),
        "mean": column.astype(float).dropna().mean(),
        "std": column.astype(float).dropna().std(),
    }


class Normalizer:

    """
    A class used to scale a continuous vector
    
    Methods
    -------
    min_max(column : pandas.Series)
        Rescales data according to the min_max normalization technique https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_(min-max_normalization)
    z_score(column : pandas.Series)
        Rescales data according to the Z-score normalization technique https://en.wikipedia.org/wiki/Feature_scaling#Standardization_(Z-score_Normalization)
    mean(column : pandas.Series)
        Rescales data according to the mean normalization technique https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization
        
    """

    @staticmethod
    def min_max(column):
        temp = _get_column_properties(column)
        min_of_column = temp["min"]
        max_of_column = temp["max"]
        y_std = []
        if min_of_column == max_of_column:
            y_std = np.zeros(len(column))
        else:
            for value in column:
                y_std.append((float(value) - min_of_column) / (max_of_column - min_of_column))
        return y_std

    @staticmethod
    def z_score(column):
        temp = _get_column_properties(column)
        standard_deviation = temp["std"]
        mean = temp["mean"]
        y_std = []
        if standard_deviation == 0:
            y_std = np.zeros(len(column))
        else:
            for value in column:
                y_std.append((float(value) - mean) / standard_deviation)
        return y_std

    @staticmethod
    def mean(column):
        temp = _get_column_properties(column)
        min_of_column = temp["min"]
        max_of_column = temp["max"]
        mean = temp["mean"]
        y_std = []
        if min_of_column == max_of_column:
            y_std = np.zeros(len(column))
        else:
            for value in column:
                y_std.append((float(value) - mean) / (max_of_column - min_of_column))
        return y_std


class Distributions:
    """
    Auxiliary class to retrieve specified distribution (used to replace reflection method)
    """
    distribution_dict = {}

    def __init__(self):
        self.distribution_dict = {
            "alpha": scipy.stats.alpha,
            "anglit": scipy.stats.anglit,
            "arcsine": scipy.stats.arcsine,
            "argus": scipy.stats.argus,
            "beta": scipy.stats.beta,
            "betaprime": scipy.stats.betaprime,
            "bradford": scipy.stats.bradford,
            "burr": scipy.stats.burr,
            "burr12": scipy.stats.burr12,
            "cauchy": scipy.stats.cauchy,
            "chi": scipy.stats.chi,
            "chi2": scipy.stats.chi2,
            "cosine": scipy.stats.cosine,
            "crystalball": scipy.stats.crystalball,
            "dgamma": scipy.stats.dgamma,
            "dweibull": scipy.stats.dweibull,
            "erlang": scipy.stats.erlang,
            "expon": scipy.stats.expon,
            "exponnorm": scipy.stats.exponnorm,
            "exponweib": scipy.stats.exponweib,
            "exponpow": scipy.stats.exponpow,
            "f": scipy.stats.f,
            "fatiguelife": scipy.stats.fatiguelife,
            "fisk": scipy.stats.fisk,
            "foldcauchy": scipy.stats.foldcauchy,
            "foldnorm": scipy.stats.foldnorm,
            "genlogistic": scipy.stats.genlogistic,
            "gennorm": scipy.stats.gennorm,
            "genpareto": scipy.stats.genpareto,
            "genexpon": scipy.stats.genexpon,
            "genextreme": scipy.stats.genextreme,
            "gausshyper": scipy.stats.gausshyper,
            "gamma": scipy.stats.gamma,
            "gengamma": scipy.stats.gengamma,
            "genhalflogistic": scipy.stats.genhalflogistic,
            "gilbrat": scipy.stats.gilbrat,
            "gompertz": scipy.stats.gompertz,
            "gumbel_r": scipy.stats.gumbel_r,
            "gumbel_l": scipy.stats.gumbel_l,
            "halfcauchy": scipy.stats.halfcauchy,
            "halflogistic": scipy.stats.halflogistic,
            "halfnorm": scipy.stats.halfnorm,
            "halfgennorm": scipy.stats.halfgennorm,
            "hypsecant": scipy.stats.hypsecant,
            "invgamma": scipy.stats.invgamma,
            "invgauss": scipy.stats.invgauss,
            "invweibull": scipy.stats.invweibull,
            "johnsonsb": scipy.stats.johnsonsb,
            "johnsonsu": scipy.stats.johnsonsu,
            "kappa4": scipy.stats.kappa4,
            "kappa3": scipy.stats.kappa3,
            "ksone": scipy.stats.ksone,
            "kstwobign": scipy.stats.kstwobign,
            "laplace": scipy.stats.laplace,
            "levy": scipy.stats.levy,
            "levy_l": scipy.stats.levy_l,
            "logistic": scipy.stats.logistic,
            "loggamma": scipy.stats.loggamma,
            "loglaplace": scipy.stats.loglaplace,
            "lognorm": scipy.stats.lognorm,
            "lomax": scipy.stats.lomax,
            "maxwell": scipy.stats.maxwell,
            "mielke": scipy.stats.mielke,
            "moyal": scipy.stats.moyal,
            "nakagami": scipy.stats.nakagami,
            "ncx2": scipy.stats.ncx2,
            "ncf": scipy.stats.ncf,
            "nct": scipy.stats.nct,
            "norm": scipy.stats.norm,
            "norminvgauss": scipy.stats.norminvgauss,
            "pareto": scipy.stats.pareto,
            "pearson3": scipy.stats.pearson3,
            "powerlaw": scipy.stats.powerlaw,
            "powerlognorm": scipy.stats.powerlognorm,
            "powernorm": scipy.stats.powernorm,
            "rdist": scipy.stats.rdist,
            "rayleigh": scipy.stats.rayleigh,
            "rice": scipy.stats.rice,
            "recipinvgauss": scipy.stats.recipinvgauss,
            "semicircular": scipy.stats.semicircular,
            "skewnorm": scipy.stats.skewnorm,
            "t": scipy.stats.t,
            "trapz": scipy.stats.trapz,
            "triang": scipy.stats.triang,
            "truncexpon": scipy.stats.truncexpon,
            "truncnorm": scipy.stats.truncnorm,
            "tukeylambda": scipy.stats.tukeylambda,
            "uniform": scipy.stats.uniform,
            "vonmises": scipy.stats.vonmises,
            "vonmises_line": scipy.stats.vonmises_line,
            "wald": scipy.stats.wald,
            "weibull_min": scipy.stats.weibull_min,
            "weibull_max": scipy.stats.weibull_max
        }

    def get_dist(self, distribution):
        try:
            return self.distribution_dict[distribution]
        except KeyError:
            return getattr(scipy.stats, distribution)


class Discretizer:
    """

    Methods
        -------
        chi_squared_goodness_of_fit(data, distribution, percentile_bins)
            Applies the Person's chi-squared goodness of fit test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test

        kolmogorov_goodness_of_fit(data, distribution, outlier_removal=False)
            Applies the Kolmogorov–Smirnov goodness of fit test https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

        distribution_binning(column, number_of_bins, statistical_test="chi2", cutoff_margin=0.2, kolmogorov_opt=True,
                             normalizer="min_max", distributions=None)
            Bins the data according to the best fitting distribution (from this list https://docs.scipy.org/doc/scipy/reference/stats.html)

        chi_squared_significant(chi_squared_statistic, df)

        kolmogorov_significant(kolmogorov_statistic, size)

    """

    @staticmethod
    def chi_squared_goodness_of_fit(data, distribution, number_of_bins):
        """Applies the chi-squared goodness of fit test according to a number of categories

        Parameters
        ----------
        data : array.float
            Your continuous observed distribution
        distribution : str
            Name of continuous distribution (from this list https://docs.scipy.org/doc/scipy/reference/stats.html)
        number_of_bins : int
            Number of categories to be considered by chi-squared test

        Returns
        -------
        list
            [0] : chi-squared statistic : float
            [1] : data : array.float
            [2] : number of estimated parameters : int
        """
        # Set up distribution and get fitted distribution parameters
        percentile_bins = np.linspace(0, 100, (number_of_bins + 1))
        percentile_bins.sort()
        dist = Distributions().get_dist(distribution)
        size = len(data)
        param = dist.fit(data)
        percentile_cutoffs = np.percentile(data, percentile_bins)
        percentile_cutoffs.sort()
        observed_frequency, bins = (np.histogram(data, bins=percentile_cutoffs))
        # Get expected counts in percentile bins based on a cumulative distribution function
        percentile_cutoffs[0] = -math.inf
        percentile_cutoffs[len(percentile_cutoffs) - 1] = math.inf
        cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], scale=param[-1])
        num_estimated_parameters = len(param)-2
        cdf_fitted.sort()
        expected_frequency = []
        for bin in range(len(percentile_bins) - 1):
            expected_cdf_area = cdf_fitted[bin + 1] - cdf_fitted[bin]
            expected_frequency.append(expected_cdf_area if expected_cdf_area != 0 else 0.00000000000000001)
        # Calculate chi-squared
        ss = 0
        for i in range(len(expected_frequency)):
            expected_frequency[i] = expected_frequency[i] * size
            ss += ((observed_frequency[i] - expected_frequency[i]) * (observed_frequency[i] - expected_frequency[i])) / expected_frequency[i]
        return [ss, data, num_estimated_parameters]

    @staticmethod
    def kolmogorov_goodness_of_fit(data, distribution, outlier_removal=False):
        """Applies the Kolmogorov-Smirnov goodness of fit test according to a number of categories

        Parameters
        ----------
        data : array.float
            Your continuous observed distribution
        distribution : str
            Name of continuous distribution from https://docs.scipy.org/doc/scipy/reference/stats.html
        outlier_removal : boolean, optional
            A flag to indicate if outliers are removed or not

        Returns
        -------
        list
            [0] : Kolmogorov-Smirnov statistic : float
            [1] : data : array.float
            [2] : number of estimated parameters : int
        """
        # 5% of original points
        N5 = int(len(data) * 0.05) if outlier_removal else 1
        num_estimated_parameters = 0
        results = []
        for aux in range(0, N5):
            dist = Distributions().get_dist(distribution)
            N = len(data)
            data.sort()
            D_plus = []
            D_minus = []
            param = dist.fit(data)
            cdfvals = dist.cdf(data, *param[:-2], loc=param[-2], scale=param[-1])
            num_estimated_parameters = len(param)-2

            idx_max_d_plus = 0
            # Calculate max(i/N-Ri)
            for i in range(1, N + 1):
                x = i / N - cdfvals[i - 1]
                D_plus.append(x)
                if x > D_plus[idx_max_d_plus]:
                    idx_max_d_plus = i - 1

            idx_max_d_minus = 0
            # Calculate max(Ri-((i-1)/N))
            for i in range(1, N + 1):
                y = cdfvals[i - 1] - ((i - 1) / N)
                D_minus.append(y)
                if y > D_minus[idx_max_d_minus]:
                    idx_max_d_minus = i - 1

            if len(results) == 0:
                results = [max(D_plus[idx_max_d_plus], D_minus[idx_max_d_minus]), data.copy()]
            else:
                ks = max(D_plus[idx_max_d_plus], D_minus[idx_max_d_minus])
                if ks < results[0]:
                    results = [ks, data.copy()]

            if D_plus[idx_max_d_plus] > D_minus[idx_max_d_minus]:
                del data[idx_max_d_plus]
            else:
                del data[idx_max_d_minus]

        results.append(num_estimated_parameters)
        return results

    @staticmethod
    def distribution_discretizer(dataset, number_of_bins, statistical_test="chi2", cutoff_margin=0.2, kolmogorov_opt=False,
                                 normalizer="min_max", distributions=None, single_column_discretization=True):
        """Discretizes data according to the best fitting distribution

        Parameters
        ----------
        dataset : pandas.Dataframe
            Your continuous observed distribution
        number_of_bins : int
            Number of categories for discretization
        statistical_test : str, optional
            Name of the statistical hypothesis test to apply fitting (available tests: "chi2", "ks")
        cutoff_margin : float, optional
            Percentage margin for border values
        kolmogorov_opt : boolean, optional
            Flag to indicate if outliers should be removed
        normalizer : str, optional
            Name of the scaling technique to use (available techniques: "min_max", "mean", "z_score")
        distributions : array.String, optional
            Names of continuous distributions from https://docs.scipy.org/doc/scipy/reference/stats.html
        single_column_discretization : boolean, optional
            Flag to indicate if DI2 should consider each column to follow a distribution or the dataset as whole to follow a distribution
        Returns
        -------
        pandas.Dataframe

        """

        if distributions is None:
            distributions = ["alpha", "anglit", "arcsine", "argus", "beta", "betaprime", "bradford",
                             "cauchy", "chi", "chi2", "cosine", "crystalball",
                             "dgamma", "dweibull", "erlang", "expon", "exponnorm",
                             "f", "fatiguelife", "foldcauchy", "foldnorm",
                             "genlogistic", "gennorm", "genpareto",
                             "genextreme", "gamma", "gengamma", "genhalflogistic",
                             "gilbrat", "gumbel_r", "gumbel_l", "halfcauchy", "halflogistic",
                             "halfnorm", "hypsecant", "invgamma", "invgauss",
                             "invweibull", "kappa3",
                             "laplace", "levy", "levy_l", "logistic", "loggamma",
                             "loglaplace", "lomax", "maxwell", "mielke", "moyal", "nakagami",
                             "norm", "pareto", "pearson3",
                             "powerlaw", "lognorm", "rdist", "rayleigh", "rice",
                             "semicircular", "t", "trapz", "triang",
                             "truncexpon", "uniform",
                             "wald", "weibull_min", "weibull_max"]
        y_std = []
        if not single_column_discretization:
            for column in dataset.columns:
                y_std.extend(_get_normalized_data(dataset[column], normalizer))

            best_dist, best_statistic, data_used = Discretizer.best_distribution(distributions, kolmogorov_opt,
                                                                                 number_of_bins, statistical_test,
                                                                                 y_std)

            for column in dataset.columns:
                dataset[column] = Discretizer.discritize(best_dist, cutoff_margin, data_used, dataset[column],
                                                        number_of_bins, _get_normalized_data(dataset[column], normalizer))
        else:
            for column in dataset.columns:
                print("variable: " + column)
                y_std = _get_normalized_data(dataset[column], normalizer)

                best_dist, best_statistic, data_used = Discretizer.best_distribution(distributions, kolmogorov_opt,
                                                                                     number_of_bins, statistical_test,
                                                                                     y_std)

                dataset[column] = Discretizer.discritize(best_dist, cutoff_margin, data_used, dataset[column], number_of_bins, y_std)

        return dataset

    @staticmethod
    def discritize(best_dist, cutoff_margin, data_used, dataset, number_of_bins, y_std):
        # Set up bins for discretization
        percentile_bins = np.linspace(0, 100, (number_of_bins + 1))
        percentile_bins.sort()
        # Fit distribution according to column
        dist = Distributions().get_dist(best_dist)
        param = dist.fit(data_used)
        new_percentile_bins = [percentile_bins[0]]
        for i in range(1, len(percentile_bins) - 1):
            if 0.0 < cutoff_margin < 0.5:
                get_boundary = (percentile_bins[i] - percentile_bins[i - 1]) * cutoff_margin
                new_percentile_bins.append((percentile_bins[i] - get_boundary) / 100)
            new_percentile_bins.append(percentile_bins[i] / 100)
            if 0.0 < cutoff_margin < 0.5:
                get_boundary = (percentile_bins[i + 1] - percentile_bins[i]) * cutoff_margin
                new_percentile_bins.append((percentile_bins[i] + get_boundary) / 100)
        new_percentile_bins.append((percentile_bins[len(percentile_bins) - 1]) / 100)
        # Find cutoff points for given distribution
        cutoff_points = []
        for i in range(0, len(new_percentile_bins)):
            cutoff_points.append(dist.ppf(new_percentile_bins[i], *param[:-2], loc=param[-2], scale=param[-1]))
        # Discretize according to cutoff points with noise tolerance
        new_column = []
        index = -1
        for column_val_index in range(0, dataset.size):
            value = dataset[column_val_index]
            if pd.isna(value):
                new_column.append(np.nan)
                continue
            index += 1
            value = y_std[index]
            if value <= cutoff_points[1]:
                new_column.append(0)
            elif value >= cutoff_points[len(cutoff_points) - 2]:
                new_column.append(len(percentile_bins) - 2)
            else:
                curr_category = 0.0
                if cutoff_margin > 0.0:
                    aux = 0
                    for i in range(1, len(cutoff_points) - 1):
                        if value <= cutoff_points[i]:
                            new_column.append(curr_category)
                            break
                        if curr_category.is_integer():
                            curr_category += 0.5
                            aux = 0
                        elif aux == 1:
                            aux = 0
                            curr_category += 0.5
                        else:
                            aux += 1
                else:
                    for i in range(1, len(cutoff_points) - 1):
                        if value < cutoff_points[i]:
                            new_column.append(curr_category)
                            break
                        else:
                            curr_category += 1
        return new_column

    @staticmethod
    def best_distribution(distributions, kolmogorov_opt, number_of_bins, statistical_test, y_std):
        dist_list = []
        for distribution in distributions:
            results = []
            if statistical_test == "chi2":
                if kolmogorov_opt:
                    results = Discretizer.kolmogorov_goodness_of_fit(y_std.copy(), distribution, kolmogorov_opt)
                    results = Discretizer.chi_squared_goodness_of_fit(results[1], distribution, number_of_bins)
                else:
                    results = Discretizer.chi_squared_goodness_of_fit(y_std.copy(), distribution, number_of_bins)
            elif statistical_test == "ks":
                results = Discretizer.kolmogorov_goodness_of_fit(y_std.copy(), distribution, kolmogorov_opt)

            dist_list.append({"distribution": distribution, "statistic": statistical_test, "statistic_value": results[0], "data": results[1], "num_estimated_parameters": results[2]})

        best_dist = ""
        best_statistic = data_used = -1
        is_significant = False
        for value in dist_list:
            significant = False
            if value["statistic"] == "chi2":
                significant = Discretizer.chi_squared_significant(value["statistic_value"], number_of_bins-1-value["num_estimated_parameters"])
            else:
                significant = Discretizer.kolmogorov_significant(value["statistic_value"], len(value["data"]))

            if significant and best_statistic == -1:
                best_dist = value["distribution"]
                best_statistic = value["statistic_value"]
                data_used = value["data"]
                is_significant = True
            elif significant and best_statistic > value["statistic_value"]:
                best_dist = value["distribution"]
                best_statistic = value["statistic_value"]
                data_used = value["data"]
                is_significant = True
        if best_statistic == -1:
            for value in dist_list:
                if best_statistic == -1:
                    best_dist = value["distribution"]
                    best_statistic = value["statistic_value"]
                    data_used = value["data"]
                elif best_statistic > value["statistic_value"]:
                    best_dist = value["distribution"]
                    best_statistic = value["statistic_value"]
                    data_used = value["data"]
        
        
        print(statistical_test)
        print(best_statistic)
        print("best distribution:" + best_dist)
        print("passes statistical test:" + str(is_significant))
        
        return best_dist, best_statistic, data_used

    @staticmethod
    def chi_squared_significant(chi_squared_statistic, df):
        if df < 1:
            return False
        else:
            return chi_squared_statistic < chi_squared_table[0.05][df]

    @staticmethod
    def kolmogorov_significant(kolmogorov_statistic, size):
        if size > 40:
            return kolmogorov_statistic < 1.36 / math.sqrt(size)
        else:
            return kolmogorov_statistic < kolmogorov_table[0.05][size]
