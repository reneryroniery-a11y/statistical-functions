# Dev:      Renery Carvalho
# Data:     on going
# Contact:  reneryroniery@gmail.com / rnyc@novonordisk.com

import math
from typing import Union, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import zscore
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os
import json
import sqlite3
import pyodbc
import warnings
"""warnings.filterwarnings('ignore')"""

#===========> Data extraction, loading and transformation
def extract_data(file_path, **kwargs):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Import the data from numerous extensions and converts into a pandas dataframe.
    
    Supported formats:
        >> .csv
        >> .xls
        >> .xlsx
        >> .xlsm
        >> .json
        >> .sql
        >> .sqlite
        >> .sqlite3
        >> .db
        >> .parquet
        >> .feather
        >> .pkl
        >> .txt
        >> .xml
        >> .h5
        >> .hdf5
    
    =============================> PARAMETERS <================================
    ---------------------------------------------------------------------------
        >> file_path (str): Path to the data file or database connection string.
        >> **kwargs: Additional arguments for specific file formats
    ---------------------------------------------------------------------------

    =============================> OUTPUT <====================================
    ---------------------------------------------------------------------------
        >> df (pd.DataFrame): Dataframe containing the extracted data.
    ---------------------------------------------------------------------------
        """
    # Defining dictionary of supported readers for different file formats
    readers = {
        '.csv': pd.read_csv,
        '.xls': pd.read_excel,
        '.xlsx': pd.read_excel,
        '.xlsm': pd.read_excel,
        '.json': pd.read_json,
        '.sql': pd.read_sql,
        '.sqlite': pd.read_sql_query,
        '.sqlite3': pd.read_sql_query,
        '.db': pd.read_sql_query,
        '.parquet': pd.read_parquet,
        '.feather': pd.read_feather,
        '.pkl': pd.read_pickle,
        '.txt': pd.read_csv, # assuming the data is in csv format
        '.xml': pd.read_xml,
        '.h5': pd.read_hdf,
        '.hdf5': pd.read_hdf
    }

    try:
        file_extension = os.path.splitext(file_path)[1].lower() # extract format

        if not os.path.exists(file_path): # Checks if file exists
            raise FileNotFoundError(f"Error > File not found: {file_path}")
        
        if file_extension in readers:
            if file_extension in ['.xls', '.xlsx', '.xlsm']: # for excel
                df = readers[file_extension](file_path, **kwargs)
            elif file_extension in ['.json']: # for json
                df = readers[file_extension](file_path, orient='records', **kwargs)
            elif file_extension in ['.sql', '.sqlite', '.sqlite3', '.db']: # special handling of sql databases or queries
                if file_extension in ['.db', '.sqlite', '.sqlite3']: # if query
                    if 'table_name' not in kwargs: # check if parameter table_name is provided
                        raise ValueError("Please provide parameter 'table_name' for SQL databases")       
                    else:
                        conn = sqlite3.connect(file_path)
                        df = pd.read_sql_query(f"SELECT * FROM {kwargs['table_name']}", conn)
                        conn.close()
                else: # if .sql
                    required_parameters = ['server','database','table_name']
                    if not all (param in kwargs for param in required_parameters):
                        raise ValueError(f"Please provide all required parameters for sql databases: {required_parameters}")
                    conn_str = (f"DRIVER={{SQL Server}};"
                                f"SERVER={kwargs['server']};"
                                f"DATABASE={kwargs['database']};"
                                )
                    if 'username' in kwargs and 'password' in kwargs: # add authentication parameters if necessary
                        conn_str += f"UID={kwargs['username']}; PWD={kwargs['password']};"
                    else:
                        conn_str += "Trusted_Connection=yes;"
                    
                    conn = pyodbc.connect(conn_str)
                    df = pd.read_sql_query(f"SELECT * FROM {kwargs['table_name']}", conn)
                    conn.close()                   
            else:
                df = readers[file_extension](file_path, **kwargs)
        else:
            raise ValueError(f"File format not supported: {file_extension}")
        
        if df.empty:
            print("Warning: loaded dataframe is empty!")

        df = df.iloc[:,1:] # dropping first column
        print(f"\nDataset successfully loaded from {file_path}")
        print(f"\n===> Shape: {df.shape}")
        print(f"\n===> First rows: ")
        print(df.head(10))
        print("\n===> Data Info: ")
        print(df.info)

        return df
    except Exception as e:
        print(f"Error loading the dataset: {str(e)}")
        raise

#===========> Transforming data
def transform_data(df):
    """
    Execute transformations on the dataframe features:
    >> check and transform data types
    >> check and corrects data integrity
    >> clean "undesired" values
    """
    pass

#===========> Descriptive Analysis
def basic_statistics(df):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Calculates the basic descriptive statistics and gives a summary of the dataset.

    >> Preview the dataset
    >> Print statistical summary of features
    >> Generate basic univariate and multivariate visualizations
    >> Analyze data types and missing values
    >> Calculate correlations between numerical features
    >> Store, save and print the results

    =============================> PARAMETERS <=================================
    ----------------------------------------------------------------------------
        >> df (pandas.DataFrame) : dataframe to be described
    
    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
        >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    results = {}

    # Basic preview
    print("\n=====> Basic Dataset Information <=====")
    print(f"\n===> Shape: {df.shape}")
    print(f"\n===> Data types: ")
    print(df.dtypes)

    results['shape'] = df.shape
    results['data_types'] = df.dtypes

    # Data integrity
    print("\n=====> Data Integrity Analysis <=====")
    integrity_summary = pd.DataFrame(
        {
            'Missing Values':df.isna().sum(),
            'Percentage Missing':(df.isna().sum()*100/len(df)).round(2)
        }
    )

    print("\n===> Missing Values Summary: ")
    print(integrity_summary)

    results['missing_values'] = integrity_summary

    # Basic statistics
    print("\n=====> Basic Statistical Summary <=====")
    print(df.describe(include='all').T)

    print("\n=====> Detailed Statistical Summary <=====")
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Additional statistics for numerical features
    if len(numerical_columns) > 0:
        print("\n===> Numerical Features Summary")
        numerical_stats = df[numerical_columns].describe().T
        print(numerical_stats)
        
        print("\n===> Additional Statistics ===")
        additional_stats = pd.DataFrame(
            {
            'Skewness': df[numerical_columns].skew(),
            'Kurtosis': df[numerical_columns].kurtosis(),
            'Variance': df[numerical_columns].var(),
            'Standard Deviation': df[numerical_columns].std(),
            'IQR': df[numerical_columns].quantile(0.75) - df[numerical_columns].quantile(0.25),
            'CV': df[numerical_columns].std() / df[numerical_columns].mean(),
            '0th central moment':stats.moment(df[numerical_columns], order=0),
            '1st central moment':stats.moment(df[numerical_columns], order=1),
            '2nd central moment':stats.moment(df[numerical_columns], order=2),
            '3rd central moment':stats.moment(df[numerical_columns], order=3)/(df[numerical_columns].std())**3,
            '4th central moment':stats.moment(df[numerical_columns], order=4)/(df[numerical_columns].std())**4
            }
        )
        print(additional_stats)
        # Skewness:             indicates relative tailedness and orientation, evaluated with Fisher's definition, i.e. minus 3
        # Kurtosis:             indicates tail weights
        # Variance:             indicates quadratic dispersion
        # Std dev:              indicates normalized dispersion
        # IQR:                  interquartile range, i.e. Q3 minus Q1
        # CV:                   coefficient of variation, indicates the extent of variability in a dataset realtive to its mean
        # 0th central moment:   it is the total mass of a probability mass/density function of a discrete/continuous random variable - it is trivial to verify that it always have value of 1
        # 1st central moment:   L1 describes location, mean - must be 0 because it is the displacement of the mean reltive to its own mean (because the moment is centralized around the mean, thus why it is called
        #                       central moment)
        # 2nd central moment:   L2 describes scale, variance - specifies the spreadness of the pmf/pdf
        # 3rd central moment:   L3 describes relative tailednes and skewness orientation, must be standardized to not be affected by location or scale - it quantifies the relative size of the two tails of a distribution
        #                       the sign indicates which tail is heavier and magnitude indicates how much
        # 4th central moment:   L4 describes tail weight, kurtosis - captures the absolute size of the two tails, evaluated with
        #                       Pearson's definition, i.e. normal => 3

    results['numerical_stats'] = numerical_stats
    results['additional_stats'] = additional_stats
    
    # Additional statistics for categorical features
    categorical_columns = df.select_dtypes(include=['object','category']).columns

    if len(categorical_columns) > 0:
        print("\n=====> Categorical Features Summary <=====")

        cat_summary = pd.DataFrame(
            {
            'Unique Values':df[categorical_columns].nunique(),
            'Most Common':df[categorical_columns].apply(lambda x: x.mode()[0] if not x.mode().empty else None),
            'Frequency':df[categorical_columns].apply(lambda x: x.value_counts().iloc[0] if not x.value_counts().empty else None),
            }
        )

    # Basic univariate and multivariate visualizations
    if len(numerical_columns) > 0:
        n_cols = min(3, len(numerical_columns))
        n_rows = (len(numerical_columns) + n_cols - 1) // n_cols

        # numerical features univariate distribution
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_columns):
            plt.subplot(n_rows, n_cols, i+1)
            sns.histplot(df[col], kde=True, bins=30) # histogram plot with kernel density estimate
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')

        # numerical features boxplots
        plt.figure(figsize=(15, 5*n_rows))
        for i, col in enumerate(numerical_columns):
            plt.subplot(n_rows, n_cols, i+1)
            sns.boxplot(data=df, y=col, color='skyblue')
            plt.title(f"Boxplot of {col}")
            plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    # categorical features univariate distribution
    if len(categorical_columns) > 0:
        plot_columns = [col for col in categorical_columns 
                        if df[col].nunique() <= 10]
        if len(plot_columns) > 0:
            n_cols = min(2, len(plot_columns))
            n_rows = (len(plot_columns) + n_cols - 1) // n_cols

            plt.figure(figsize=(15, 5*n_rows))
            for i, col in enumerate(plot_columns):
                plt.subplot(n_rows, n_cols, i+1)
                df[col].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')

            plt.show()

    return results

#===========> Verifying normality Analysis
def normality_tests(df, methodology='all', alpha=0.05):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform normality tests on the dataset features distributions following the defined method
    and significance levelgiven the assumptions are true and print the results and their 
    interpretation.

    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------
    
    >> df (pandas.DataFrame) : dataframe that will undergo the normality tests

    >> methodology (str or list), default='all' : defines which methods will be used in the following list
        >> Shapiro-Wilk : 'sw'
            - data is continuous
            - independent observations
            - n < 50, but valid up to n ~ 2000
            - high power for detecting non-normality
            - does not work well with samples that have many identical values

        >> Kolmogorov-Smirnov  : 'ks'
            - generalist but less powerful than S-W
            - Apply Lilliefors correction if 'mu'&'var' are extracted from the sample instead of the population
            - Sensitive to deviations in the centre of the distribution
            - May be preferred for very large samples, i.e. n > 2000
            - data is continuous

        >> Anderson-Darling : 'ad'
            - emphasises the tails of the distribution, recommended for heavier tails
            - more sensitive than K-S when detecting deviations in the tails
            - recommended when 50 <= n <= 300

        >> D'Agostino and Pearson's : 'dp'
            - best for moderate to large samples, i.e. n >
            - combines skewness and kurtosis to assess normality
            - good for detecting asymmetry and peakedness
            - recommended when 50 <= n <= 300

        >> Jarque-Bera : 'jb'
            - best for large samples
            - also based on skewness and kurtosis
            - very common in econometrics and time series analysis
            - recommended when n > 300

        >> QQ Plot : 'qq'
            - graphical method to assess normality
            - non-parametric test
            - powerful visual tool
            - use to visually assess how well the data follows a normal distribution
            - recommended when n > 300

    >> alpha (float), default=0.05 : significance level for the tests

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results for each feature
    ----------------------------------------------------------------------------
    """
    results = {}

    # selecting numerical features only
    numerical_features = df.select_dtypes(include=[np.number]).columns
    if len(numerical_features) == 0:
        raise ValueError("No numerical features found in the dataset.")
    
    # converting methodology parameter to lowercase and string
    if isinstance(methodology, str):
        methodology = [methodology.lower()]
    methodology = [m.lower() for m in methodology]

    if 'all' in methodology:
        methodology = ['sw', 'ks', 'ad', 'dp', 'jb', 'qq']

    print("\n=====> Normality Tests Results <=====")
    for feature in numerical_features:
        data = df[feature].dropna() # drop missing values
        n = len(data) # sample size
        results[feature] = {} # create a key for each feature results

        print(f"\n===> Feature: {feature}")
        print(f"Sample Size: {n}")

        if 'sw' in methodology:
            if n < 2000:
                statistic, p_value = stats.shapiro(data)
                results[feature]['shapiro-wilk'] = {
                    'statistic': statistic,
                    'p_value': p_value,
                    'interpretation': ('Normal' if p_value > alpha else 'Not Normal')
                }
                print("\n>> Shapiro-Wilk Test")
                print(f"\nStatistic: {results[feature]['shapiro-wilk']['statistic']:.4f}, "
                      f"\nP-value: {results[feature]['shapiro-wilk']['p_value']:.4f}, "
                      f"\nInterpretation: {results[feature]['shapiro-wilk']['interpretation']}"
                      )
            else:
                print("Shapiro-Wilk test is not recommended for sample size >= 2000. Skipping...")

        if 'ks' in methodology:
            data_standardized = (data - np.mean(data)) / np.std(data)
            statistic, p_value = stats.kstest(data_standardized, 'norm')
            results[feature]['kolmogorov-smirnov'] = {
                'statistic': statistic,
                'p_value': p_value,
                'interpretation': ('Normal' if p_value > alpha else 'Not Normal')
            }
            print("\n>> Kolmogorov-Smirnov Test")
            print(f"\nStatistic: {results[feature]['kolmogorov-smirnov']['statistic']:.4f}, "
                  f"\nP-value: {results[feature]['kolmogorov-smirnov']['p_value']:.4f}, "
                  f"\nInterpretation: {results[feature]['kolmogorov-smirnov']['interpretation']}"
                  )
            
        if 'ad' in methodology:
            statistic, critical_values, significance_level = stats.anderson(data, dist='norm')
            p_value = 1 - stats.norm.cdf(statistic)
            results[feature]['anderson-darling'] = {
                'statistic': statistic,
                'critical_values': critical_values,
                'significance_level': significance_level,
                'p_value': p_value,
                'interpretation': ('Normal' if statistic < critical_values[2] else 'Not Normal')
            }
            print("\n>> Anderson-Darling Test")
            print(f"\nStatistic: {results[feature]['anderson-darling']['statistic']:.4f}, "
                  f"\nCritical Values: {results[feature]['anderson-darling']['critical_values']}, "
                  f"\nSignificance Level: {results[feature]['anderson-darling']['significance_level']}, "
                  f"\nP-value: {results[feature]['anderson-darling']['p_value']:.4f}, "
                  f"\nInterpretation: {results[feature]['anderson-darling']['interpretation']}"
                  )

        if 'dp' in methodology:
            statistic, p_value = stats.normaltest(data)
            results[feature]['dagostino-pearson'] = {
                'statistic': statistic,
                'p_value': p_value,
                'interpretation': ('Normal' if p_value > alpha else 'Not Normal')
            }
            print("\n>> D'Agostino and Pearson's Test")
            print(f"\nStatistic: {results[feature]['dagostino-pearson']['statistic']:.4f}, "
                  f"\nP-value: {results[feature]['dagostino-pearson']['p_value']:.4f}, "
                  f"\nInterpretation: {results[feature]['dagostino-pearson']['interpretation']}"
                  )

        if 'jb' in methodology:
            statistic, p_value = stats.jarque_bera(data)
            results[feature]['jarque-bera'] = {
                'statistic': statistic,
                'p_value': p_value,
                'interpretation': ('Normal' if p_value > alpha else 'Not Normal')
            }
            print("\n>> Jarque-Bera Test")
            print(f"\nStatistic: {results[feature]['jarque-bera']['statistic']:.4f}, "
                  f"\nP-value: {results[feature]['jarque-bera']['p_value']:.4f}, "
                  f"\nInterpretation: {results[feature]['jarque-bera']['interpretation']}"
                  )
            
        if 'qq' in methodology:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            stats.probplot(data, dist="norm", plot=ax1)
            ax1.set_title(f"Q-Q Plot for {feature}")
            ax1.set_xlabel('Theoretical Quantiles')
            ax1.set_ylabel('Sample Quantiles')

            # Plotting the histogram with KDE
            sns.histplot(data=data, kde=True, ax=ax2)
            ax2.set_title(f"Histogram with KDE for {feature}")
            ax2.set_xlabel(feature)
            ax2.set_ylabel('Density')

            # add normal curve
            x = np.linspace(data.min(), data.max(), 100) # generate x values based on sample range
            y = stats.norm.pdf(x, data.mean(), data.std()) # generate y values based on normal distribution w/ sample parameters
            ax2.plot(x, y*len(data)*(data.max()-data.min())/30, color='red', label='Normal Curve') # plot normal curve
            ax2.legend()
            plt.show()
        else:
            raise ValueError("Error: The selected methodology does not exists in this function")

    return results

#===========> Correlation Analysis
def correlation_analysis(df, target, methodology='pearson'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Analyses the Pearson's correlation values for the dataset features with and without considering
    the outliers, plots both matrixes and print the interpretation of the results regarding regression,
    multicolinearity, dimensionality reduction.
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------
    
    >> df (pandas.DataFrame) : dataframe that will undergo the calculations

    >> method (str) : name of the method for correlation evaluation between the default pearson, 
    spearman or kendall

    >> target (str) : name of the target feature

    >> outliers (bool) : defines if consider or not consider the fatures' outliers, default is False

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results for each feature
    ----------------------------------------------------------------------------
    """

    results = {}
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # Check if selected methodology exists
    if methodology not in ['pearson', 'kendall', 'spearman']:
        raise ValueError("Error! Method must be 'pearson', 'kendall' or 'spearman")

    # Check if target feature is numerical
    if target not in numerical_columns.tolist():
        raise ValueError(f"Error! Target feature '{target}' must be numerical.")

    print("\n=====> Correlation Analysis w/ Outliers <=====")
    # Correlation analysis if numerical features exists
    if len(numerical_columns) > 1:
        print(f"\n===> {methodology.capitalize()} Correlation Matrix")
        correlation_matrix = df[numerical_columns].corr(method=methodology)
        print(correlation_matrix)

        # Plotting the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"{methodology.capitalize()} Correlation Matrix (w/ outliers)")
        plt.show()

        results['corr_outliers'] = correlation_matrix

    print("\n=====> Correlation Analysis w/o Outliers <=====")
    # Removing outliers with Z-score method
    df_wo_outliers = df[(np.abs(np.zscore(df[numerical_columns])) < 3).all(axis=1)]
    if len(numerical_columns) > 1:
        print(f"\n===> {methodology.capitalize()} Correlation Matrix")
        correlation_matrix_wo = df_wo_outliers[numerical_columns].corr(method=methodology)
        print(correlation_matrix_wo)

        plt.figure(fisize=(10, 8))
        sns.heatmap(correlation_matrix_wo, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title(f"{methodology.capitalize()} Correlation Matrix (w/o outliers)")
        plt.show()

        results['corr_no_outliers'] = correlation_matrix_wo

    # Scatter-plots to verify non-linearities
    if target in numerical_columns:
        print(f"\n=====> Scatter-Plots <=====")
        sns.pairplot(df[numerical_columns])
        plt.show()

    # Interpreting the results
    if methodology == 'pearson':
        print("\n=====> Interpretation of Pearson's Correlation Results <=====")
        target_correlations = correlation_matrix[target].drop(index=target)
        print("\n===> Target feature '{target}' correlations: ")
        print(target_correlations.sort_values(ascending=False))

        # strengths
        print("\n===> Strenght of relationships")
        for feature, corr in target_correlations.items():
            if abs(corr) > 0.7:
                print(f"'{feature}' has **STRONG** correlation with target '{target}' : |r| = {corr:.2f}")
            elif 0.3 <= abs(corr) <= 0.7:
                print(f"'{feature}' has **MODERATE** correlation with target '{target}' : |r| = {corr:.2f}")
            else:
                print(f"'{feature}' has **WEAK** correlation with target '{target}' : |r| = {corr:.2f}")
        
        # multicolinearity
        print("\n===> Redundancy, Multicolinearity and Dimensionality Reduction Assessment : |r| > 0.8")
        high_corr_pairs = []
        for feature1 in numerical_columns:
            for feature2 in numerical_columns:
                if feature1 != feature2 and abs(correlation_matrix.loc[feature1, feature2] > 0.8):
                    print(f"'{feature1}' and '{feature2}' shows **HIGH MULTICOLINEARITY** : |r| = {correlation_matrix.loc[feature1, feature2]:.2f}")
                    print(f"\nConsider **DROPPING**, **COMBINING** or **REDUCING** '{feature1}'+'{feature2}'.")
                    high_corr_pairs.append((feature1, feature2))
                else:
                    print(f"No highly redundant realtionship found between '{feature1}' and '{feature2}'")

        # strong predictors for regression models
        print("\n===> Strong Predictors for Regression Models")
        strong_predictors = target_correlations[target_correlations.abs() > 0.7].index.tolist()
        if strong_predictors:
            print(f"Features with strong predictive power for '{target}': {', '.join(strong_predictors)}")
        else:
            print(f"No features with strong predictive power for '{target}' found.")

        return {"correlation_matrix_with_outliers": correlation_matrix,
                "correlation_matrix_without_outliers": correlation_matrix_wo,
                "target_correlations": target_correlations,
                "high_corr_pairs": high_corr_pairs,
                "strong_predictors": strong_predictors
                }
    
#===========> Outliers Detection
def outliers_detection(df, methodology='z-score'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Detects numerical outliers in the dataset features using the defined method, store them in arrays
    and print the results.
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> df (pandas.DataFrame) : dataframe that will undergo the calculations

    >> methodology (str) : name of the method for detecting outliers between

        > Z-Score
            - detects outliers based on the number of standard deviations from the mean
            - suitable for univariate and normally distributed data
            - easy to interpret

        > IQR (Interquartile Range)
            - detects outliers based on the range between the first and third quartiles
            - robust to skewed distributions
            - easy to interpret

        > LOF (Local Outlier Factor)
            - detects outliers based on the local density of data points
            - suitable for multivariate and high-dimensional data
            - recommended for small to medium-sized datasets

        > KNN (K-Nearest Neighbors)
            - detects outliers based on the distance to the k-nearest neighbors
            - suitable for multivariate and high-dimensional data
            - recommended for small to medium-sized datasets
            - easy to interpret
        
        > Isolation Forest
            - detects outliers based on the isolation of data points in a random forest
            - suitable for multivariate and high-dimensional data
            - recommended for large datasets
            - good for scalability and efficiency
        
        > DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
            - detects outliers based on the density of data points in a region
            - used in clustered data
            - suitable for multivariate and high-dimensional data
            - good for scalability and efficiency
            - recommended for large datasets
    
        > One-Class SVM (Support Vector Machine)
            - detects outliers based on the support vector machine algorithm
            - suitable for multivariate and high-dimensional data
            - recommended for large datasets

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the outliers indices for each feature
    ----------------------------------------------------------------------------
    """
    results = {}

    # selecting numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns

    # checking methodology
    if methodology not in ['z-score', 'iqr', 'lof', 'knn', 'isolation_forest', 'dbscan', 'one_class_svm']:
        raise ValueError(f"Error! Invalid method {methodology}. Please select from 'z-score', 'iqr', 'lof', 'knn', 'isolation_forest', 'dbscan', 'one_class_svm'")

    print(f"\n=====> Outliers Detection with method {methodology} <=====")

    # Z-score
    if methodology == 'z-score':
        for col in numerical_columns:
            z_scores = zscore(df[col])
            outliers = np.where(np.abs(z_scores) > 3)[0]
            results[col] = outliers
            print(f"\nZ-Score -> Count of feature '{col}' outliers: {len(outliers)}")
            print(f"Outliers Indices: {outliers}")
    
    # Interquartile Range
    elif methodology == 'iqr':
        for col in numerical_columns:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5*iqr
            upper_bound = q3 + 1.5*iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            results[col] = outliers

    # Local Outlier Factor
    elif methodology == 'lof':
        lof = LocalOutlierFactor(n_neighbors=20)
        y_pred = lof.fit_predict(df[numerical_columns])
        outliers = np.where(y_pred == -1)[0] # outliers are evaluated as -1
        results['LOF'] = outliers
        print(f"\nLOF -> outliers: {len(outliers)}")
        print(f"Outliers Indices: {list(outliers)}")

    # K-nearest neighbors
    elif methodology == 'knn':
        knn = NearestNeighbors(n_neighbors=5)
        knn.fit(df[numerical_columns])
        distances, _ = knn.kneighbors(df[numerical_columns])
        threshold = np.percentiles(np.distance[:, 4], 95) # 95th percentile as cutoff
        outliers = np.where(distances[:,4] > threshold)[0]
        results['KNN'] = outliers
        print(f"\nKNN -> outliers: {len(outliers)}")
        print(f"Outliers Indices: {list(outliers)}")

    # Isolation Forest
    elif methodology == 'isolation_forest':
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        y_pred = iso_forest.fit_predict(df[numerical_columns])
        outliers = np.where(y_pred == -1)[0] # outliers are evaluated as -1
        results['Isolation Forest'] = outliers
        print(f"\nIsolation Forest -> outliers: {len(outliers)}")
        print(f"Outliers Indices: {list(outliers)}")

    # DBSCAN
    elif methodology == 'dbscan':
        dbscan = DBSCAN(eps=1.5, min_samples=5)
        y_pred = dbscan.fit_predict(df[numerical_columns])
        outliers = np.where(y_pred == -1) # outliers are evaluated as -1
        results['DBSCAN'] = outliers
        print(f"\nDBSCAN -> outliers: {len(outliers)}")
        print(f"Outliers Indices: {list(outliers)}")

    elif methodology == 'one_class_svm':
        oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
        y_pred = oc_svm.fit_predict(df[numerical_columns])
        outliers = np.where(y_pred == -1) # outliers are evaluated as -1
        results['One-Class SVM'] = outliers
        print(f"\nOne-Class SVM -> outliers: {len(outliers)}")
        print(f"Outliers Indices: {list(outliers)}")        

    return results

#===========> Probabilistic Analysis
def probability_distributions(df): # (NOT IMPLEMENTED)
    """
    Fit the dataset features to various probability distributions, print the results and plot the
    distributions.

    >> Normal Distribution (c.r.v)
        - symmetric, bell-shaped distribution
        - defined by mean and standard deviation

    >> Exponential Distribution (c.r.v)
        - used for modeling time until an event occurs
        - defined by rate parameter

    >> Poisson Distribution (d.r.v)
        - used for modeling count data
        - defined by rate parameter

    >> Binomial Distribution (d.r.v)
        - used for modeling binary outcomes
        - defined by number of trials and probability of success
    
    >> Bernoulli Distribution (d.r.v)
        - used for modeling binary outcomes
        - defined by probability of success
        - special case of the binomial distribution with one trial

    >> Discrete Uniform Distribution (d.r.v)
        - all outcomes are equally likely
        - defined by minimum and maximum values

    >> Log-Normal Distribution (c.r.v)
        - used for modeling positively skewed data
        - defined by mean and standard deviation of the logarithm of the variable

    >> Weibull Distribution (c.r.v)
        - used for modeling life data and reliability analysis
        - defined by shape and scale parameters

    >> Beta Distribution
        - used for modeling proportions and probabilities
        - defined by two shape parameters
    """
    pass

#===========> Hypothesis Testing and Confidence Intervals
# Parametric tests
def test_population_mean(sample_data, population_mean, population_std=None, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform hypothesis testing for population mean using Z-test or T-test based on the assumptions
    and parameters provided.
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> sample_data (array like) : dataframe that will undergo the calculations

    >> population_mean (numerical) : hypothesized population mean to perform the test

    >> population_std (numerical) : standard deviation of the population, default is 'None'

    >> alpha (numerical) : significance level, default is 0.05 for 95% confidence

    >> alternative (str) : type of test, options are:
        > 'two-sided' (default)
            H₀: μ = μ₀ (population true mean = 'population_mean')
            H₁: μ ≠ μ₀ (population true mean ≠ 'population_mean)
        > 'less'
            H₀: μ ≥ μ₀
            H₁: μ < μ₀
        > 'greater'
            H₀: μ ≤ μ₀
            H₁: μ > μ₀

    =============================> METHODOLOGIES <===============================
    -----------------------------------------------------------------------------
    >> Z-Test
        - use when population std is known
        - recommended for large samples, i.e. when n > 30

    >> T-Test
        - use when population std is unknown
        - recommended for small samples, i.e. when n < 30

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    # normality check
    normality_check = normality_tests(sample_data)

    # Check dimension of sample data
    if isinstance(sample_data, pd.DataFrame):
        if sample_data.shape[1] != 1:
            raise ValueError("Error! Sample data must be 1D array like.")
        sample_data = sample_data.iloc[:,0].values
    elif isinstance(sample_data, pd.Series):
        sample_data = sample_data.values

    n = len(sample_data)
    sample_mean = np.mean(sample_data)
    confidence_level = 1 - alpha
    results = {}

    # input validation
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Error! Alternative must be 'two-sided', 'less' or 'greater'.")
    if not 0 < alpha < 1:
        raise ValueError("Error! Significance level alpha must be between 0 and 1.")
    if population_std is not None and population_std <= 0:
        raise ValueError("Error! Population standard deviation must be positive.")
    if population_std is not None and n < 30:
        warnings.warn("Warning! Z-test is not recommended for small samples (n < 30) unless population std is known.", UserWarning)

    # hypothesis testing
    if population_std is not None: # Z-test
        z_stat = (sample_mean - population_mean)/(population_std/np.sqrt(n))

        if alternative == 'two-sided':
            p_value = 2*(1 - stats.norm.cdf(abs(z_stat)))
            z_critical = stats.norm.ppf(1 - alpha/2)
        elif alternative == 'less':
            p_value = stats.norm.cdf(z_stat)
            z_critical = stats.norm.ppf(alpha)
        else:
            p_value = 1 - stats.norm.cdf(z_stat)
            z_critical = stats.norm.ppf(1 - alpha)

        margin_error = z_critical*(population_std / np.sqrt(n))
        if alternative == 'two-sided':
            ci_upper = sample_mean + margin_error
            ci_lower = sample_mean - margin_error
        elif alternative == 'less':
            ci_upper = float('inf')
            ci_lower = sample_mean - margin_error
        else:
            ci_upper = sample_mean + margin_error
            ci_lower = float('-inf')
        
        test_name = "Z-test"
        stat_name = "z_statistic"
        stat = z_stat
        if alternative == 'two-sided':
            if p_value < alpha:
                conclusion = f"Reject Ho : μ = {population_mean} and Accept H₁: μ ≠ {population_mean}" 
            else: 
                conclusion = f"Fail to reject Ho : μ = {population_mean} and Reject H₁: μ ≠ {population_mean}"
        elif alternative == 'less':
            if p_value < alpha:
                conclusion = f"Reject Ho : μ ≥ {population_mean} and Accept H₁: μ < {population_mean}"
            else:
                conclusion = f"Fail to reject Ho : μ ≥ {population_mean} and Reject H₁: μ < {population_mean}"
        else:
            if p_value < alpha:
                conclusion = f"Reject Ho : μ ≤ {population_mean} and Accept H₁: μ > {population_mean}"
            else:
                conclusion = f"Fail to reject Ho : μ ≤ {population_mean} and Reject H₁: μ > {population_mean}"

    else: # t-test
        t_stat = stats.ttest_1samp(sample_data, population_mean).statistic

        if alternative == 'two-sided':
            p_value = 2*(1 - stats.t.cdf(abs(t_stat), df=n-1))
            t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
        elif alternative == 'less':
            p_value = stats.t.cdf(t_stat, df=n-1)
            t_critical = stats.t.ppf(alpha, df=n-1)
        else:
            p_value = 1 - stats.t.cdf(t_stat, df=n-1)
            t_critical = stats.t.ppf(1 - alpha, df=n-1)

        sample_std = np.std(sample_data, ddof=1)
        margin_error = t_critical*(sample_std / np.sqrt(n))
        if alternative == 'two-sided':
            ci_upper = sample_mean + margin_error
            ci_lower = sample_mean - margin_error
        elif alternative =='less':
            ci_upper = float('inf')
            ci_lower = sample_mean - margin_error
        else:
            ci_upper = sample_mean + margin_error
            ci_lower = float('-inf')
        
        test_name = "T-test"
        stat_name = "t_statistic"
        stat = t_stat
        if alternative == 'two-sided':
            if p_value < alpha:
                conclusion = f"Reject Ho : μ = {population_mean} and Accept H₁: μ ≠ {population_mean}" 
            else: 
                conclusion = f"Fail to reject Ho : μ = {population_mean} and Reject H₁: μ ≠ {population_mean}"
        elif alternative == 'less':
            if p_value < alpha:
                conclusion = f"Reject Ho : μ ≥ {population_mean} and Accept H₁: μ < {population_mean}"
            else:
                conclusion = f"Fail to reject Ho : μ ≥ {population_mean} and Reject H₁: μ < {population_mean}"
        else:
            if p_value < alpha:
                conclusion = f"Reject Ho : μ ≤ {population_mean} and Accept H₁: μ > {population_mean}"
            else:
                conclusion = f"Fail to reject Ho : μ ≤ {population_mean} and Reject H₁: μ > {population_mean}"

    # Plot confidence interval
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))
    ax1.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence intervals')
    ax1.plot([sample_mean], [1], 'ro', label='sample mean')
    ax1.axvline(x=sample_mean, color='r', linestyle='--', alpha=0.5, label='sample mean')
    ax1.axvline(x=population_mean, color='g', linestyle=':', alpha = 0.5, label='hypothetical mean')
    ax1.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5, label='lower bound')
    ax1.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5, label='upper bound')
    ax1.set_title(f"Confidence Interval for Population Mean ({confidence_level}%)")
    plt.xlabel('Mean')
    plt.legend()
    plt.show()

    # Test visualization
    x = np.linspace(-4, 4, 1000)
    if population_std is not None:
        dist = stats.norm.pdf(x)
    else:
        dist = stats.t.pdf(x, df=n-1)

    ax2.plot(x, dist, 'b-', label='Null distribution')
    
    # Shading rejection region(s)
    if alternative == 'two-sided':
        critical = stats.norm.ppf(1-alpha/2) if population_std is not None else stats.t.ppf(1-alpha/2, df=n-1)
        ax2.fill_between(x[x <= -critical], dist[x <= -critical], color='r', alpha=0.3)
        ax2.fill_between(x[x >= critical], dist[x >= critical], color='r', alpha=0.3)
    elif alternative == 'less':
        critical = stats.norm.ppf(alpha) if population_std is not None else stats.t.ppf(alpha, df=n-1)
        ax2.fill_between(x[x <= critical], dist[x <= critical], color='r', alpha=0.3)
    else:
        critical = stats.norm.ppf(1-alpha) if population_std is not None else stats.t.ppf(1-alpha, df=n-1)
        ax2.fill_between(x[x >= critical], dist[x >= critical], color='r', alpha=0.3)

    # Plot test statistic
    test_stat = z_stat if population_std is not None else t_stat
    ax2.axvline(test_stat, color='g', linestyle='--', label='Test Statistic')
    ax2.set_title(f"Hypothesis Test Visualization")
    ax2.legend()

    plt.show()

    results[test_name] = {
        'test_name':test_name,
        'test_statistic':stat,
        'sample_mean':sample_mean,
        'confidence_interval':(ci_lower, ci_upper),
        'p_value':p_value,
        'normality':normality_check,
        'conclusion':conclusion
    }
    print(results)

    return results

def test_population_proportion(sample_data, hypothetical_proportion, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform hypothesis testing for population proportion using Z-test based on the assumptions
    and parameters provided. The 'sample_data' must contain successes and failures evaluated
    as 1 and 0 respectively
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> df (array) : dataframe that will undergo the calculations

    >> hypothetical_proportion (numerical) : hypothetical proportion of the population

    >> alpha (numerical) : significance level, default is 0.05 for 95% confidence

    >> alternative (str) : type of test, options are:
        > 'two-sided' (default)
            H₀: p = ^p (population true proportion = 'hypothetical_proportion')
            H₁: p ≠ ^p (population true proportion ≠ 'hypothetical_proportion')
        > 'less'
            H₀: p ≥ ^p
            H₁: p < ^p
        > 'greater'
            H₀: p ≤ ^p
            H₁: p > ^p

    =============================> METHODOLOGIES <===============================
    -----------------------------------------------------------------------------
    >> Z-Test
        - use when population std is known
        - recommended for large samples, i.e. when n > 30, OR for normal distributions

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    # normality check
    normality_check = normality_tests(sample_data)

    # Check dimension of sample data
    if isinstance(sample_data, pd.DataFrame):
        if sample_data.shape[1] != 1:
            raise ValueError("Error! Sample data must be 1D array like.")
        sample_data = sample_data.iloc[:,0].values
    elif isinstance(sample_data, pd.Series):
        sample_data = sample_data.values
    
    n = len(sample_data)
    successes = np.sum(sample_data) / n # successes are preset as 1
    p_estimate = successes / n
    confidence_level = 1 - alpha
    results = {}

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Error! Alternatives must be 'two-sided', 'less' or 'greater'.")
    if not 0 <= hypothetical_proportion <= 1:
        raise ValueError("Error! Hypothesized proportion must be between 0 and 1.")
    if n*hypothetical_proportion < 5 or n*(1-hypothetical_proportion) < 5:
        warnings.warn(f"Sample size is {n} and might be too small for normal approximation.")

    # Z-statistic calculation
    std_error = np.sqrt(hypothetical_proportion*(1-hypothetical_proportion)/n)
    ci_std_error = np.sqrt((p_estimate*(1 - p_estimate))/n)
    z_stat = (p_estimate - hypothetical_proportion)/std_error

    # critical values and p_value calculations
    if alternative == 'two-sided':
        p_value = 2*(1-stats.norm.cdf(abs(z_stat)))
        z_critical = stats.norm.ppf(1 - alpha/2)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(alpha)
    else:
        p_value = 1 - stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(1 - alpha)

    margin_error = z_critical*ci_std_error
    # Confidence interval
    if alternative == 'two-sided':
        ci_lower = p_estimate - margin_error
        ci_upper = p_estimate + margin_error
    elif alternative == 'less':
        ci_lower = p_estimate - margin_error
        ci_upper = 1
    else:
        ci_lower = 0
        ci_upper = p_estimate + margin_error

    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = f"Reject H₀: p = {hypothetical_proportion} and Accept H₁: p ≠ {hypothetical_proportion}"
        else:
            conclusion = f"Fail to reject H₀: p = {hypothetical_proportion} and Reject H₁: p ≠ {hypothetical_proportion}"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = f"Reject H₀: p ≥ {hypothetical_proportion} and Accept H₁: p < {hypothetical_proportion}"
        else:
            conclusion = f"Fail to reject H₀: p ≥ {hypothetical_proportion} and Reject H₁: p < {hypothetical_proportion}"
    else:
        if p_value < alpha:
            conclusion = f"Reject H₀: p ≤ {hypothetical_proportion} and Accept H₁: p > {hypothetical_proportion}"
        else:
            conclusion = f"Fail to reject H₀: p ≤ {hypothetical_proportion} and Reject H₁: p > {hypothetical_proportion}"

    # Plot intervals
    plt.figure(figsize=(10, 2))
    plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
    plt.plot([p_estimate], [1], 'ro', label='sample proportion')
    plt.axvline(x=hypothetical_proportion, color='g', linestyle='--', alpha=0.5, label='hypothesized proportion')
    plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5, label='lower bound')
    plt.axvline(x=ci_upper, color='b', linestyle=':', label='upper bound')
    plt.title(f"Confidence Interval for Population Proportion ({confidence_level*100}%)")
    plt.xlabel('Proportion')
    plt.xlim(-0.1, 1.1)
    plt.legend()
    plt.show()

    results = {
        'Z-test': {
            'test_name': 'Z-test',
            'test_statistic': z_stat,
            'proportion_estimate':p_estimate,
            'hypothetical_proportion': hypothetical_proportion,
            'sample_statistics': {
                'successes': successes,
                'failures': n - successes,
                'sample_size': n
            },
            'confidence_interval':(ci_lower, ci_upper),
            'p_value': p_value,
            'normality':normality_check,
            'conclusion': conclusion
        }
    }
    print(results)
    return results

def test_population_variance(sample_data, hypothetical_variance, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform hypothesis testing for population variance using Chi-square based on the assumptions
    and parameters provided. The function is implemented only for two-tailed tests.
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> df (array) : dataframe that will undergo the calculations

    >> hypothetical_proportion (numerical) : hypothetical variance of the population

    >> alpha (numerical) : significance level, default is 0.05 for 95% confidence

    >> alternative (str) : type of test, options are:
        > 'two-sided' (default)
            H₀: σ² = σ₀² (population true variance = 'hypothetical_variance')
            H₁: σ² ≠ σ₀² (population true proportion ≠ 'hypothetical_variance')
        > 'less'
            H₀: σ² ≥ σ₀²
            H₁: σ² < σ₀²
        > 'greater'
            H₀: σ² ≤ σ₀²
            H₁: σ² > σ₀²

    =============================> METHODOLOGIES <===============================
    -----------------------------------------------------------------------------
    >> Chi-square
        - use when population std is known
        - recommended for large samples, i.e. when n > 30, OR for normal distributions

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    # normality check
    normality_check = normality_tests(sample_data)

    # Check dimension of sample data
    if isinstance(sample_data, pd.DataFrame):
        if sample_data.shape[1] != 1:
            raise ValueError("Error! Sample data must be 1D array like.")
        sample_data = sample_data.iloc[:,0].values
    elif isinstance(sample_data, pd.Series):
        sample_data = sample_data.values

    n = len(sample_data)
    sample_var = np.var(sample_data, ddof=1) # successes are preset as 1
    confidence_level = 1 - alpha
    results = {}

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Error! Alternative must be 'two-sided', 'less' or 'greater'.")
    
    # Chi-square test for variance
    test_name = 'Chi2-test'
    chi2_stat = (n - 1)*sample_var / hypothetical_variance
    if alternative == 'two-sided':
        p_value = 2*min(stats.chi2.cdf(chi2_stat, n-1), 1 - stats.chi2.cdf(chi2_stat, n-1))
        crit_lower = stats.chi2.ppf(alpha/2, n-1)
        crit_upper = stats.chi2.ppf(1 - alpha/2, n-1)
        ci_lower = (n - 1)*sample_var / stats.chi2.ppf(1 - alpha/2, n-1)
        ci_upper = (n - 1)*sample_var / stats.chi2.ppf(alpha/2, n-1)
    elif alternative == 'less':
        p_value = stats.chi2.cdf(chi2_stat, n-1)
        crit_lower = stats.chi2.ppf(alpha, n-1)
        crit_upper = float('inf')
        ci_lower = 0
        ci_upper = (n - 1)*sample_var / stats.chi2.ppf(alpha, n-1)
    else:
        p_value = 1 - stats.chi2.cdf(chi2_stat, n-1)
        crit_lower = 0
        crit_upper = stats.chi2.ppf(1 - alpha, n-1)
        ci_lower = (n - 1)*sample_var / stats.chi2.ppf(1 - alpha/2, n-1)
        ci_upper = float('inf')

    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = f"Reject H₀: σ² = {hypothetical_variance} and Accept H₁: σ² ≠ {hypothetical_variance}"
        else:
            conclusion = f"Fail to reject H₀: σ² = {hypothetical_variance} and Reject H₁: σ² ≠ {hypothetical_variance}"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = f"Reject H₀: σ² ≥ {hypothetical_variance} and Accept H₁: σ² < {hypothetical_variance}"
        else:
            conclusion = f"Fail to reject H₀: σ² ≥ {hypothetical_variance} and Reject H₁: σ² < {hypothetical_variance}"
    else:
        if p_value < alpha:
            conclusion = f"Reject H₀: σ² ≤ {hypothetical_variance} and Accept H₁: σ² > {hypothetical_variance}"
        else:
            conclusion = f"Fail to reject H₀: σ² ≤ {hypothetical_variance} and Reject H₁: σ² > {hypothetical_variance}"

    # Plot sample distribution with variance indication
    fig, (ax1, ax2) = plt.subplots (1, 2, figsize=(12,4))

    sns.histplot(data=sample_data, kde=True, ax=ax1)
    ax1.axvline(np.mean(sample_data), color='r', linestyle='--', label='Mean')
    ax1.axvline(np.mean(sample_data) + float(np.sqrt(sample_var)), color='g', 
                linestyle=':', label='±1 SD')
    ax1.axvline(np.mean(sample_data) - float(np.sqrt(sample_var)), color='g', 
                linestyle=':')
    ax1.set_title('Sample Distribution with Variance Indication')
    ax1.legend()

    # Sample variance distribution visualization
    x_range = np.linspace(0, stats.chi2.ppf(0.999, n-1), 1000)
    chi2_pdf = stats.chi2.pdf(x_range, n-1)
    ax2.plot(x_range, chi2_pdf, 'b-', label='Chi-square distribution')
    ax2.axvline(float(chi2_stat), color='r', linestyle='--', 
                label='Test statistic')
    
    # Shade rejection region based on alternative
    if alternative == 'two-sided':
        if crit_lower > 0:
            ax2.fill_between(x_range[x_range <= crit_lower], 
                           chi2_pdf[x_range <= crit_lower], 
                           color='r', alpha=0.3)
        ax2.fill_between(x_range[x_range >= crit_upper], 
                        chi2_pdf[x_range >= crit_upper], 
                        color='r', alpha=0.3)
    elif alternative == 'less':
        if crit_lower > 0:
            ax2.fill_between(x_range[x_range <= crit_lower], 
                           chi2_pdf[x_range <= crit_lower], 
                           color='r', alpha=0.3)
    else:
        ax2.fill_between(x_range[x_range >= crit_upper], 
                        chi2_pdf[x_range >= crit_upper], 
                        color='r', alpha=0.3)
    
    ax2.set_title(f"Chi-square Distribution (df={n-1})")
    ax2.legend()
    plt.tight_layout()
    plt.xlim()
    plt.show()

    # Plot confidence intervals
    ci_lower = float(ci_lower) if isinstance(ci_lower, (pd.Series, pd.DataFrame)) else ci_lower
    ci_upper = float(ci_upper) if isinstance(ci_upper, (pd.Series, pd.DataFrame)) else ci_upper
    sample_var = float(sample_var) if isinstance(sample_var, (pd.Series, pd.DataFrame)) else sample_var

    plt.figure(figsize=(10, 2))
    if np.isfinite(ci_lower) and np.isfinite(ci_upper):
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
    elif np.isfinite(ci_lower):
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
        plt.arrow(ci_lower, 1, 1, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
    else:
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
        plt.arrow(ci_upper, 1, -1, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
    
    plt.plot([sample_var], [1], 'ro', label='sample variance')
    plt.axvline(x=float(sample_var), color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=hypothetical_variance, color='g', linestyle='--', alpha=0.5, label='hypothetical variance')
    plt.title(f"Confidence Interval for Population Variance ({confidence_level*100}%)")
    plt.xlabel('Variance')
    plt.xlim()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    results = {
            'test_name': test_name,
            'test_statistic': chi2_stat,
            'variance_estimate':sample_var,
            'hypothetical_variance': hypothetical_variance,
            'confidence_interval':(ci_lower, ci_upper),
            'p_value': p_value,
            'critical_values': (crit_lower, crit_upper),
            'normality':normality_check,
            'conclusion': conclusion
        }
    
    return results

def test_2independent_mean(sample1, sample2, pop_std1=None, pop_std2=None, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Performs hypothesis testing for difference between means of two independent populations.
    
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------
    >> sample1, sample2 : array-like
        Sample data from two independent populations
    
    >> pop_std1, pop_std2 : float, optional
        Known population standard deviations (if available)
    
    >> alpha : float, default=0.05
        Significance level
    
    >> alternative : str, default='two-sided'
    
    >> Type of test: 'two-sided', 'less', or 'greater'
        
        > Two-sided:
            H₀: μ₁ = μ₂
            H₁: μ₁ ≠ μ₂
        > Less:
            H₀: μ₁ ≥ μ₂
            H₁: μ₁ < μ₂
        > Greater:
            H₀: μ₁ ≤ μ₂
            H₁: μ₁ > μ₂
    
    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict : Results containing test statistics and conclusions
    """
    # Check normality assumption
    print("\n=== Normality Tests ===")
    norm_test1 = normality_tests(sample1)
    print("\nSample 1 Normality Test:")
    print(f"Results: {norm_test1}")
    
    norm_test2 = normality_tests(sample2)
    print("\nSample 2 Normality Test:")
    print(f"Results: {norm_test2}")

    # Check dimension and type of samples data
    if isinstance(sample1, pd.DataFrame):
        if sample1.shape[1] != 1:
            raise ValueError("Error! Sample 1 must be 1D array like.")
        sample1 = sample1.iloc[:,0].values
    elif isinstance(sample1, pd.Series):
        sample1 = sample1.values
    if isinstance(sample2, pd.DataFrame):
        if sample2.shape[1] != 1:
            raise ValueError("Error! Sample 2 must be 1D array like.")
        sample2 = sample2.iloc[:,0].values
    elif isinstance(sample2, pd.Series):
        sample2 = sample2.values
    
    # Calculate basic statistics
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = np.mean(sample1), np.mean(sample2)
    var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)
    std1, std2 = np.std(sample1, ddof=1), np.std(sample2, ddof=1)
    
    # Test for equality of variances (Levene's test)
    levene_stat, levene_p = stats.levene(sample1, sample2)
    print("\n=== Equality of Variances Test (Levene) ===")
    print(f"Test statistic: {levene_stat:.4f}")
    print(f"P-value: {levene_p:.4f}")
    equal_variances = levene_p > alpha
    if equal_variances:
        print("Variances are equal! Proceed with pooled variance t-test.")
    else:
        print("Variances are NOT equal! Proceed with Welch's corrected t-test.")
    
    # Determine test type and calculate test statistic
    if pop_std1 is not None and pop_std2 is not None:
        # Z-test (known population variances)
        print("\nUsing Z-test (known population variances)")
        se = np.sqrt((pop_std1**2/n1) + (pop_std2**2/n2))
        stat = (mean1 - mean2) / se
        df = float('inf')
        test_type = 'Z-test'
    else:
        # T-test
        if equal_variances:
            # Pooled variance t-test
            print("\nUsing pooled variance t-test (equal variances)")
            pooled_var = ((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            stat = (mean1 - mean2) / se
            df = n1 + n2 - 2
            test_type = 'pooled t-test'
        else:
            # Welch's t-test
            print("\nUsing Welch's t-test (unequal variances)")
            se = np.sqrt(var1/n1 + var2/n2)
            stat = (mean1 - mean2) / se
            # Welch–Satterthwaite equation for degrees of freedom
            df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
        test_type = "Welch's T-test"
    
    # Calculate p-value based on alternative hypothesis
    if test_type == 'Z-test':
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.norm.cdf(abs(stat)))
        elif alternative == 'less':
            p_value = stats.norm.cdf(stat)
        else:  # alternative == 'greater'
            p_value = 1 - stats.norm.cdf(stat)
    else:  # T-test
        if alternative == 'two-sided':
            p_value = 2 * (1 - stats.t.cdf(abs(stat), df))
        elif alternative == 'less':
            p_value = stats.t.cdf(stat, df)
        else:  # alternative == 'greater'
            p_value = 1 - stats.t.cdf(stat, df)
    
    # Calculate confidence interval
    confidence_level = 1 - alpha
    if test_type == 'Z-test':
        z_critical = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_critical * se
    else:
        t_critical = stats.t.ppf((1 + confidence_level) / 2, df)
        margin_error = t_critical * se
    
    ci_lower = (mean1 - mean2) - margin_error
    ci_upper = (mean1 - mean2) + margin_error
    
    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sample distributions
    sns.histplot(data=sample1, kde=True, ax=ax1, alpha=0.5, label='Sample 1')
    sns.histplot(data=sample2, kde=True, ax=ax1, alpha=0.5, label='Sample 2')
    ax1.axvline(mean1, color='r', linestyle='--', label='Mean 1')
    ax1.axvline(mean2, color='b', linestyle='--', label='Mean 2')
    ax1.set_title('Samples Distributions with Means Indication')
    ax1.legend()
    
    # Test statistic distribution
    x = np.linspace(-4, 4, 1000)
    if test_type == 'Z-test':
        pdf = stats.norm.pdf(x)
    else:
        pdf = stats.t.pdf(x, df)
    
    ax2.plot(x, pdf)
    ax2.axvline(stat, color='r', linestyle='--', label='Test statistic')
    
    # Shade rejection region(s)
    if alternative == 'two-sided':
        if test_type == 'Z-test':
            critical = stats.norm.ppf(1 - alpha/2)
        else:
            critical = stats.t.ppf(1 - alpha/2, df)
        ax2.fill_between(x[x <= -critical], pdf[x <= -critical], color='r', alpha=0.3)
        ax2.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)
    elif alternative == 'less':
        if test_type == 'Z-test':
            critical = stats.norm.ppf(alpha)
        else:
            critical = stats.t.ppf(alpha, df)
        ax2.fill_between(x[x <= critical], pdf[x <= critical], color='r', alpha=0.3)
    else:  # alternative == 'greater'
        if test_type == 'Z-test':
            critical = stats.norm.ppf(1 - alpha)
        else:
            critical = stats.t.ppf(1 - alpha, df)
        ax2.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)
    
    ax2.set_title(f'{test_type} Distribution')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Plot confidence interval
    plt.figure(figsize=(10, 2))
    plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
    plt.plot([mean1 - mean2], [1], 'ro', label='means difference')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='origin')
    plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5, label='lower bound')
    plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5, label='upper bound')
    plt.title(f"Confidence Interval for Difference in Means ({confidence_level*100}%)")
    plt.legend()
    plt.show()
    
    # Determine conclusion
    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = "Reject H₀ : μ₁ = μ₂ and Accept H₁ : μ₁ ≠ μ₂"
        else:
            conclusion = "Fail to reject H₀ : μ₁ = μ₂ and Reject H₁ : μ₁ ≠ μ₂"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = "Reject H₀ : μ₁ ≥ μ₂ and Accept H₁ : μ₁ < μ₂"
        else:
            conclusion = "Fail to reject H₀ : μ₁ ≥ μ₂ and Reject H₁ : μ₁ < μ₂"
    else:  # alternative == 'greater'
        if p_value < alpha:
            conclusion = "Reject H₀ : μ₁ ≤ μ₂ and Accept H₁ : μ₁ > μ₂"
        else:
            conclusion = "Fail to reject H₀ : μ₁ ≤ μ₂ and Reject H₁ : μ₁ > μ₂"
    
    return {
        'test_type': test_type,
        'test_statistic': stat,
        'degrees_of_freedom': df,
        'p_value': p_value,
        'mean_difference': mean1 - mean2,
        'confidence_interval': (ci_lower, ci_upper),
        'equal_variances': equal_variances,
        'normality_tests': {
            'sample1': norm_test1,
            'sample2': norm_test2
        },
        'variance_test': {
            'statistic': levene_stat,
            'p_value': levene_p
        },
        'sample_statistics': {
            'sample1': {
                'size': n1,
                'mean': mean1,
                'variance': var1,
                'std': std1
            },
            'sample2': {
                'size': n2,
                'mean': mean2,
                'variance': var2,
                'std': std2
            }
        },
        'conclusion': conclusion
    }

def test_2paired_mean(before, after, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Performs hypothesis testing for difference between means of two paired samples.
    
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------
    >> before, after : array-like
        Sample data before and after treatment
    
    >> alpha : float, default=0.05
        Significance level
    
    >> alternative : str, default='two-sided'
    
    >> Type of test: 'two-sided', 'less', or 'greater'
        
        > Two-sided:
            H₀: μ_d = 0 (no change)
            H₁: μ_d ≠ 0
        > Less:
            H₀: μ_d ≥ 0 (increased mean or no change)
            H₁: μ_d < 0
        > Greater:
            H₀: μ_d ≤ 0 (decreased mean or no change)
            H₁: μ_d > 0
    
    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict : Results containing test statistics and conclusions
    """
    results = {}

    # Calculate differences and basic statistics
    differences = after - before


    # Check normality assumption
    print("\n=== Normality Tests ===")
    norm_test_before = normality_tests(before)
    print("\nBefore Treatment Normality Test:")
    print(f"Results: {norm_test_before}")
    
    norm_test_after = normality_tests(after)
    print("\nAfter Treatment Normality Test:")
    print(f"Results: {norm_test_after}")

    norm_test_diff = normality_tests(differences)
    print("\nDifferences Normality Test:")
    print(f"Results: {norm_test_diff}")

    # Check dimension and type of samples data
    if isinstance(before, pd.DataFrame):
        if before.shape[1] != 1:
            raise ValueError("Error! Before sample must be 1D array like.")
        before = before.iloc[:,0].values
    elif isinstance(before, pd.Series):
        before = before.values
    if isinstance(after, pd.DataFrame):
        if after.shape[1] != 1:
            raise ValueError("Error! After sample must be 1D array like.")
        after = after.iloc[:,0].values
    elif isinstance(after, pd.Series):
        after = after.values

    differences = after - before
    n = len(differences) 
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    se_diff = std_diff / np.sqrt(n)

    # t-statistic, p_value and confidence interval calculations
    t_stat = mean_diff / se_diff

    if alternative == 'two-sided':
        p_value = 2*(1 - stats.t.cdf(abs(t_stat), n-1))
    elif alternative == 'less':
        p_value = stats.t.cdf(t_stat, n-1)
    else:
        p_value = 1 - stats.t.cdf(t_stat, n-1)
    
    confidence_level = 1 - alpha
    t_critical = stats.t.ppf((1+confidence_level)/2, n-1)
    margin_error = t_critical*se_diff
    ci_lower = mean_diff - margin_error
    ci_upper = mean_diff + margin_error

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    # Before x After scatter plot
    ax1.scatter(before, after, alpha=0.5)
    min_val = min(np.min(before), np.min(after))
    max_val = max(np.max(before), np.max(after))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='y=x line')
    ax1.set_xlabel('Before Treatment')
    ax1.set_ylabel('After Treatment')
    ax1.set_title('Before vs After')

    # differences histogram
    sns.histplot(differences, kde=True, ax=ax2, color='orange', label='Differences')
    ax2.axvline(mean_diff, color='r', linestyle='--', label='Mean Difference')
    ax2.set_title('Differences Distribution')
    ax2.legend()

    # Q-Q plot for differences
    stats.probplot(differences, dist="norm", plot=ax3)
    ax3.set_title('Q-Q Plot of Differences')

    # Test statistic distribution
    x = np.linspace(-4, 4, 1000)
    pdf = stats.t.pdf(x, n-1)
    ax4.plot(x, pdf, 'b-', label='t-distribution')
    ax4.axvline(t_stat, color='r', linestyle='--', label='Test statistic')

    # shading rejection region(s)
    if alternative == 'two-sided':
        critical = stats.t.ppf(1 - alpha/2, n-1)
        ax4.fill_between(x[x <= -critical], pdf[x <= -critical], color='r', alpha=0.3)
        ax4.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)
    elif alternative == 'less':
        critical = stats.t.ppf(alpha, n-1)
        ax4.fill_between(x[x <= critical], pdf[x <= critical], color='r', alpha=0.3)
    else:
        critical = stats.t.ppf(1 - alpha, n-1)
        ax4.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)

    ax4.set_title(f't-distribution (df={n-1})')
    ax4.legend()

    plt.show()

    # Plot confidence interval
    plt.figure(figsize=(10, 2))
    plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
    plt.axvline(x=mean_diff, color='r', linestyle='--', label='Mean Difference')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Origin')
    plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5, label='Lower Bound')
    plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5, label='Upper Bound')
    plt.title(f"Confidence Interval for Mean Difference ({confidence_level*100}%)")
    plt.xlabel('Mean Difference')
    plt.legend()
    plt.show()

    # test conclusion
    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = "Reject H₀ : μ_d = 0 and Accept H₁ : μ_d ≠ 0"
        else:
            conclusion = "Fail to reject H₀ : μ_d = 0 and Reject H₁ : μ_d ≠ 0"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = "Reject H₀ : μ_d ≥ 0 and Accept H₁ : μ_d < 0"
        else:
            conclusion = "Fail to reject H₀ : μ_d ≥ 0 and Reject H₁ : μ_d < 0"
    else: 
        if p_value < alpha:
            conclusion = "Reject H₀ : μ_d ≤ 0 and Accept H₁ : μ_d > 0"
        else:
            conclusion = "Fail to reject H₀ : μ_d ≤ 0 and Reject H₁ : μ_d > 0"

    results = {
        'test_type': 'Paired T-test',
        'test_statistic': t_stat,
        'degrees_of_freedom': n-1,
        'p_value': p_value,
        'mean_difference': mean_diff,
        'confidence_interval': (ci_lower, ci_upper),
        'normality_tests': {
            'before': norm_test_before,
            'after': norm_test_after,
            'differences': norm_test_diff
        },
        'sample_statistics': {
            'before': {
                'size': len(before),
                'mean': np.mean(before),
                'std': np.std(before, ddof=1)
            },
            'after': {
                'size': len(after),
                'mean': np.mean(after),
                'std': np.std(after, ddof=1)
            },
            'differences': {
                'size': n,
                'mean': mean_diff,
                'std': std_diff
            }
        },
        'conclusion': conclusion
    }
    return results

def test_2independent_proportion(sample1, sample2, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform hypothesis testing for two independent population proportions using Z-test 
    based on the assumptions and parameters provided. The samples must contain successes 
    and failures evaluated as 1 and 0 respectively
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> sample1, sample2 (array like) : samples that will undergo the calculations

    >> hypothetical_proportion (numerical) : hypothetical proportion of the population

    >> alpha (numerical) : significance level, default is 0.05 for 95% confidence

    >> alternative (str) : type of test, options are:
        > 'two-sided' (default)
            H₀: p₁ = p₂
            H₁: p₁ ≠ p₂
        > 'less'
            H₀: p₁ ≥ p₂
            H₁: p₁ < p₂
        > 'greater'
            H₀: p₁ ≤ p₂
            H₁: p₁ > p₂

    =============================> METHODOLOGIES <===============================
    -----------------------------------------------------------------------------
    >> Z-Test
        - use when population std is known
        - recommended for large samples, i.e. when n > 30, OR for normal distributions

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    results = {}
    # normality check
    normality1 = normality_tests(sample1)
    normality2 = normality_tests(sample2)
    
    # Check dimension of sample data
    if isinstance(sample1, pd.DataFrame):
        if sample1.shape[1] != 1:
            raise ValueError("Error! Samples must be 1D array like")
        sample1 = sample1.iloc[:,0].values
    elif isinstance(sample1, pd.Series):
        sample1 = sample1.values

    if isinstance(sample2, pd.DataFrame):
        if sample2.shape[1] != 1:
            raise ValueError("Error! Samples must be 1D array like")
        sample2 = sample2.iloc[:,0].values
    elif isinstance(sample2, pd.Series):
        sample2 = sample2.values

    n1 = len(sample1)
    n2 = len(sample2)

    successes1 = np.sum(sample1) / n1
    successes2 = np.sum(sample2) / n2

    p_sample1 = successes1 / n1
    p_sample2 = successes2 / n2

    confidence_level = 1 - alpha

    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Alternative hypothesis must be 'two-sided', 'less', or 'greater'")
    if not (0 < alpha < 1):
        raise ValueError("Alpha must be between 0 and 1")
    if (n1* p_sample1 < 5 or n1*(1 - p_sample1) < 5) or (n2* p_sample2 < 5 or n2*(1 - p_sample2) < 5):
        raise ValueError(f"Sample 1 size {n1} or sample 2 size {n2} are too small and might not be normally distributed. Use Fisher's exact test instead.")

    # z-statistic calculation
    p_pooled = (successes1 + successes2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    z_stat = (p_sample1 - p_sample2) / se

    # critical values and p_values
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        z_critical = stats.norm.ppf(1 - alpha/2)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(alpha)
    else:
        p_value = 1 - stats.norm.cdf(z_stat)
        z_critical = stats.norm.ppf(1 - alpha)

    margin_error = z_critical*se
    # confidence interval calculation
    if alternative == 'two-sided':
        ci_lower = (p_sample1 - p_sample2) - margin_error
        ci_upper = (p_sample1 - p_sample2) + margin_error
    elif alternative == 'less':
        ci_lower = (p_sample1 - p_sample2) - margin_error
        ci_upper = 1
    else:
        ci_lower = 0
        ci_upper = (p_sample1 - p_sample2) + margin_error

    # Visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Sample distributions
    sns.histplot(data=sample1, kde=True, ax=ax1, alpha=0.5, label='Sample 1')
    sns.histplot(data=sample2, kde=True, ax=ax1, alpha=0.5, label='Sample 2')
    ax1.axvline(p_sample1, color='r', linestyle='--', label='Sample 1 proportion')
    ax1.axvline(p_sample2, color='b', linestyle='--', label='Sample 2 proportion')
    ax1.set_title('Samples Distributions with Proportions Indication')
    ax1.legend()

    # Test statistic distribution
    x = np.linspace(-4, 4, 1000)
    pdf = stats.norm.pdf(x)
    ax2.plot(x, pdf, 'b-', label='Test statistic Normal distribution')
    ax2.axvline(z_stat, color='r', linestyle='--', label='Test statistic')

    # Shade rejection region(s)
    if alternative == 'two-sided':
        critical = stats.norm.ppf(1 - alpha/2)
        ax2.fill_between(x[x <= -critical], pdf[x <= -critical], color='r', alpha=0.3)
        ax2.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)
    elif alternative == 'less':
        critical = stats.norm.ppf(alpha)
        ax2.fill_between(x[x <= critical], pdf[x <= critical], color='r', alpha=0.3)
    else:
        critical = stats.norm.ppf(1 - alpha)
        ax2.fill_between(x[x >= critical], pdf[x >= critical], color='r', alpha=0.3)

    ax2.set_title(f'Z-test Distribution (α={alpha})')
    ax2.legend()
    plt.tight_layout()
    plt.show()

    # Plot confidence interval
    plt.figure(figsize=(10, 2))
    plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
    plt.axvline(x=p_sample1 - p_sample2, color='r', linestyle='--', label='Sample Proportions Difference')
    plt.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Origin')
    plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5, label='Lower Bound')
    plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5, label='Upper Bound')
    plt.title(f"Confidence Interval for Difference in Proportions ({confidence_level*100}%)")
    plt.xlabel('Difference in Proportions')
    plt.legend()
    plt.show()

    # conclusion
    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = "Reject H₀ : p₁ = p₂ and Accept H₁ : p₁ ≠ p₂"
        else:
            conclusion = "Fail to reject H₀ : p₁ = p₂ and Reject H₁ : p₁ ≠ p₂"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = "Reject H₀ : p₁ ≥ p₂ and Accept H₁ : p₁ < p₂"
        else:
            conclusion = "Fail to reject H₀ : p₁ ≥ p₂ and Reject H₁ : p₁ < p₂"
    else:
        if p_value < alpha:
            conclusion = "Reject H₀ : p₁ ≤ p₂ and Accept H₁ : p₁ > p₂"
        else:
            conclusion = "Fail to reject H₀ : p₁ ≤ p₂ and Reject H₁ : p₁ > p₂"

    results = {
        'test_type': 'Z-test for Proportions',
        'test_statistic': z_stat,
        'p_value': p_value,
        'sample_proportions': {
            'sample1': p_sample1,
            'sample2': p_sample2
        },
        'confidence_interval': (ci_lower, ci_upper),
        'normality_tests': {
            'sample1': normality1,
            'sample2': normality2
        },
        'sample_statistics': {
            'sample1': {
                'size': n1,
                'successes': successes1,
                'proportion': p_sample1
            },
            'sample2': {
                'size': n2,
                'successes': successes2,
                'proportion': p_sample2
            }
        },
        'conclusion': conclusion
    }
    print(results)
    return results

def test_2population_variance(sample1, sample2, alpha=0.05, alternative='two-sided'):
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Perform hypothesis testing for population variance using F-distribution based
    on the provided parameters. Assumptions:

    -> both samples comes from populations that follows normal distribution
    -> samples are independent
   
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> sample1, sample2 (array like) : samples that will undergo the calculations

    >> alpha (numerical) : significance level, default is 0.05 for 95% confidence

    >> alternative (str) : type of test, options are:
        > 'two-sided' (default)
            H₀: σ₁² = σ₂²
            H₁: σ₁² ≠ σ₂²
        > 'less'
            H₀: σ₁² ≥ σ₂²
            H₁: σ₁² < σ₂²
        > 'greater'
            H₀: σ₁² ≤ σ₂²
            H₁: σ₁² > σ₂²

    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict: dictionary containing the results
    ----------------------------------------------------------------------------
    """
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError("Error! Alternative must be 'two-sided', 'less' or 'greater'.")
    if not 0 < alpha < 1:
        raise ValueError("Error! Alpha must be between 0 and 1.")
    
    results = {}
    
    # normality check
    normality1 = normality_tests(sample1)
    normality2 = normality_tests(sample2)
    
    # Check dimension of sample data
    if isinstance(sample1, pd.DataFrame):
        if sample1.shape[1] != 1:
            raise ValueError("Error! Samples must be 1D array like")
        sample1 = sample1.iloc[:,0].values
    elif isinstance(sample1, pd.Series):
        sample1 = sample1.values

    if isinstance(sample2, pd.DataFrame):
        if sample2.shape[1] != 1:
            raise ValueError("Error! Samples must be 1D array like")
        sample2 = sample2.iloc[:,0].values
    elif isinstance(sample2, pd.Series):
        sample2 = sample2.values

    n1 = len(sample1)
    n2 = len(sample2)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)

    # calculating F-statistic
    if var1 >= var2:
        f_stat = var1/var2
        df1, df2 = n1 - 1, n2 - 1
        larger_var = "sample 1"
    else:
        f_stat = var2/var1
        df1, df2 = n2 - 1, n1 - 1
        larger_var = "sample 2"

    # critical values, p_values and confidence interval calculations
    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.f.cdf(f_stat, df1, df2))
        crit_lower = stats.f.ppf(alpha/2, df1, df2)
        crit_upper = stats.f.ppf(1 - alpha/2, df1, df2)
        ci_lower = (var1 / var2) * (1 / stats.f.ppf(1 - alpha/2, df1, df2))
        ci_upper = (var1 / var2) * (1 / stats.f.ppf(alpha/2, df1, df2))
    elif alternative == 'less':
        p_value = stats.f.cdf(f_stat, df1, df2)
        crit_lower = stats.f.ppf(alpha, df1, df2)
        crit_upper = float('inf')
        ci_lower = 0
        ci_upper = (var1 / var2) * (1 / stats.f.ppf(alpha, df1, df2))
    else:
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        crit_lower = 0 
        crit_upper = stats.f.ppf(1 - alpha, df1, df2)
        ci_lower = (var1 / var2) * (1 / stats.f.ppf(1 - alpha, df1, df2))
        ci_upper = float('inf')

    # Conclusion
    if alternative == 'two-sided':
        if p_value < alpha:
            conclusion = f"Reject H₀ : σ₁² = σ₂² and Accept H₁ : σ₁² ≠ σ₂²"
        else:
            conclusion = f"Fail to reject H₀ : σ₁² = σ₂² and Reject H₁ : σ₁² ≠ σ₂²"
    elif alternative == 'less':
        if p_value < alpha:
            conclusion = f"Reject H₀ : σ₁² ≥ σ₂² and Accept H₁ : σ₁² < σ₂²"
        else:
            conclusion = f"Fail to reject H₀ : σ₁² ≥ σ₂² and Reject H₁ : σ₁² < σ₂²"
    else:
        if p_value < alpha:
            conclusion = f"Reject H₀ : σ₁² ≤ σ₂² and Accept H₁ : σ₁² > σ₂²"
        else:
            conclusion = f"Fail to reject H₀ : σ₁² ≤ σ₂² and Reject H₁ : σ₁² > σ₂²"

    # Samples distributions with variances and means indications
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    sns.histplot(data=sample1, kde=True, ax=ax1, alpha=0.5, label='Sample 1')
    sns.histplot(data=sample2, kde=True, ax=ax1, alpha=0.5, label='Sample 2')
    ax1.axvline(np.mean(sample1), color='r', linestyle='--', label='Sample 1 Mean')
    ax1.axvline(np.mean(sample1)+float(np.sqrt(var1)), color='g', linestyle=':', label='±1 SD Sample 1')
    ax1.axvline(np.mean(sample1)-float(np.sqrt(var1)), color='g', linestyle=':', label='±1 SD Sample 1')
    ax1.axvline(np.mean(sample2), color='b', linestyle='--', label='Sample 2 Mean')
    ax1.axvline(np.mean(sample2)+float(np.sqrt(var2)), color='y', linestyle=':', label='±1 SD Sample 2')
    ax1.axvline(np.mean(sample2)-float(np.sqrt(var2)), color='y', linestyle=':', label='±1 SD Sample 2')
    ax1.set_title('Samples Distributions with Means and Variances Indication')
    ax1.legend()

    # F-distribution plot
    x_range = np.linspace(0, stats.f.ppf(0.999, df1, df2), 1000)
    f_pdf = stats.f.pdf(x_range, df1, df2)
    ax2.plot(x_range, f_pdf, 'b-', label='F-distribution')
    ax2.axvline(f_stat, color='r', linestyle='--', label='F-statistic')

    # Shade rejection region based on alternative
    if alternative == 'two-sided':
        if crit_lower > 0:
            ax2.fill_between(x_range[x_range <= crit_lower], 
                           f_pdf[x_range <= crit_lower], 
                           color='r', alpha=0.3)
        ax2.fill_between(x_range[x_range >= crit_upper], 
                        f_pdf[x_range >= crit_upper], 
                        color='r', alpha=0.3)
    elif alternative == 'less':
        if crit_lower > 0:
            ax2.fill_between(x_range[x_range <= crit_lower], 
                           f_pdf[x_range <= crit_lower], 
                           color='r', alpha=0.3)
    else:
        ax2.fill_between(x_range[x_range >= crit_upper], 
                        f_pdf[x_range >= crit_upper], 
                        color='r', alpha=0.3)
        
    ax2.set_title(f"F-distribution (df1={df1}, df2={df2})")
    ax2.legend()
    plt.tight_layout()
    plt.xlim()
    plt.show()

    # plot confidence intervals
    ci_lower = float(ci_lower) if isinstance(ci_lower, (pd.Series, pd.DataFrame)) else ci_lower
    ci_upper = float(ci_upper) if isinstance(ci_upper, (pd.Series, pd.DataFrame)) else ci_upper
    var1 = float(var1) if isinstance(var1, (pd.Series, pd.DataFrame)) else var1
    var2 = float(var2) if isinstance(var2, (pd.Series, pd.DataFrame)) else var2

    plt.figure(figsize=(10, 2))
    if np.isfinite(ci_lower) and np.isfinite(ci_upper):
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.plot([var1/var2], [1], 'ro', label='Sample Variance Ratio')
        plt.axvline(x=1, color='g', linestyle=':', alpha=0.5, label='Null Value')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
        plt.legend()
    elif np.isfinite(ci_lower):
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.plot([var1/var2], [1], 'ro', label='Sample Variance Ratio')
        plt.axvline(x=1, color='g', linestyle=':', alpha=0.5, label='Null Value')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
        plt.arrow(ci_lower, 1, 1, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
        plt.legend()
    else:
        plt.plot([ci_lower, ci_upper], [1, 1], 'b-', linewidth=2, label='confidence interval')
        plt.plot([var1/var2], [1], 'ro', label='Sample Variance Ratio')
        plt.axvline(x=1, color='g', linestyle=':', alpha=0.5, label='Null Value')
        plt.axvline(x=ci_lower, color='b', linestyle=':', alpha=0.5)
        plt.axvline(x=ci_upper, color='b', linestyle=':', alpha=0.5)
        plt.arrow(ci_upper, 1, -1, 0, head_width=0.1, head_length=0.1, fc='b', ec='b')
        plt.legend()    

    results = {
        'test_type': 'F-test for Variance',
        'test_statistic': f_stat,
        'degrees_of_freedom': (df1, df2),
        'p_value': p_value,
        'sample_variances': {
            'sample1': var1,
            'sample2': var2
        },
        'confidence_interval': (ci_lower, ci_upper),
        'normality_tests': {
            'sample1': normality1,
            'sample2': normality2
        },
        'sample_statistics': {
            'sample1': {
                'size': n1,
                'variance': var1
            },
            'sample2': {
                'size': n2,
                'variance': var2
            }
        },
        'conclusion': conclusion
    }
    print(results)
    return results

def test_1way_anova(samples, alpha=0.05): # (IMPLEMENTATION ON GOING)

    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Performs hypothesis testing for difference between means of k independent
    populations using the unidirectional ANOVA method. Assumptions:

    ->  samples are independent with no repeated measures or paired groups
    ->  samples must approximately follows normal distribution
    ->  variances accross groups should be equal, i.e. homogeneity of variances
        or homoscedasticity

    Due to the nature of the test, there is only one pair of hypothesis to test:
            
        H₀: μ₁ = μ₂ = ... μ_k ∀ k ∈ N
        H₁: At least one μ_i ≠ μ_j ∀ (i, j) c [1, k] | i ≠ j
    
    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------
    >> samples : array-like
        dataframe or array-like containing the samples
    
    >> pop_std1, pop_std2 : float, optional
        Known population standard deviations (if available)
    
    >> alpha : float, default=0.05
        Significance level
    
    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> dict : Results containing test statistics and conclusions
    """
    # normality check for each sample
    normalities = {}
    for i, sample in enumerate(samples):
        normalities[f"group_{i}"] = normality_tests(sample)
        if (normalities[f"group_{i}"][0]['shapiro-wilk']['p_value'] < alpha or
            normalities[f"group_{i}"][0]['anderson-darling']['p_value'] < alpha or
            normalities[f"group_{i}"][0]['dagostino-pearson']['p_value'] < alpha
            ):
            normal_distribution = False
        else:
            normal_distribution = True

    # inputs validation
    if not 0 < alpha < 1:
        raise ValueError("Error! Alpha must be between 0 and 1.")
    
    for i, sample in enumerate(samples):
        if isinstance(sample, pd.DataFrame):
            if sample.shape[1] != 1:
                raise ValueError("Error! Samples must be 1D array like")
            samples[i] = sample.iloc[:,0].values
        elif isinstance(sample, pd.Series):
            samples[i] = sample.values

    homoscedasticity = {}
    # procedures considering normality results
    if not normal_distribution:
        warnings.warn("""
        Normality assumption violated! Consider using:
            1. Kruskal-Wallis H-test (non-parametric alternative)
            2. Transform your data (e.g. log transformation)
            3. Proceed with ANOVA if sample sizes are large (n > 30) due to CLT
        
        The function will proceed with Levene's test to check equality of variances
        and decide whether it follows Welch's test OR to condemn the statistical relevance
        of the current test if the homoscedasticity is false.
        """)

        # homogeneity of variances (or homoscedasticity) with Levene's test or Bartlett
        print("\n=====> Homoscedasticity Test w/ Levene's <=====")

        # levene test
        levene_stat, levene_p = stats.levene(*samples)
        homoscedasticity['levene'] = {
            'statistic': levene_stat,
            'p_value': levene_p,
            'interpretation': ('Equal Variances' if levene_p > alpha else 'Unequal Variances')
        }
        print(f"Levene's statistic = {levene_stat: .4f}")
        print(f"p_value = {levene_p: .4f}")
        print(f"interpretation: {homoscedasticity['levene']['interpretation']}")

        homoscedastic_levene = (True if homoscedasticity['levene']['p_value'] > alpha else False)
        homoscedastic_bartlett = False

        if not homoscedastic_levene:
            warnings.warn("""
            Homoscedasticity assumption violated according to Levene's test!
            Proceed with Welch's ANOVA instead.
            """)
    else: # in case samples are normally distributed, check variance equality with Bartlett
        print("\n=====> Homoscedasticity Test w/ Bartlett <=====") 
        # bartlett test
        bartlett_stat, bartlett_p = stats.bartlett(*samples)
        homoscedasticity['bartlett'] = {
            'statistic': bartlett_stat,
            'p_value': bartlett_p,
            'interpretation': ('Equal Variances' if bartlett_p > alpha else 'Unequal Variances')
        }
        print(f"Bartlett's statistic = {bartlett_stat: .4f}")
        print(f"Bartlett's p_value = {bartlett_p: .4f}")
        print(f"Interpretation: {homoscedasticity['bartlett']['interpretation']}")

        homoscedastic_bartlett = (True if homoscedasticity['bartlett']['p_value'] > alpha else False)
        homoscedastic_levene = False

        if not homoscedastic_bartlett:
            warnings.warn("""
            Homoscedasticity assumption violated according to Bartlett's test!
            Proceed with Welch's test.
        """)
      
    # in case data is normally distributed and is homoscedastic, perform standard 1way ANOVA
    if locals().get('homoscedastic_bartlett', False) == False:
        if (normal_distribution and homoscedastic_levene) == True:
            # basic statistics
            n = [len(sample) for sample in samples] # vector of samples sizes
            total_n = np.sum(n)
            mu = [np.mean(sample) for sample in samples] # vector for samples means
            var = [np.var(sample) for sample in samples] # vector of samples variances
            
            # overall mean
            mu_all = np.mean([mean*size for mean, size in zip(mu, n)]) / total_n

            # sum of squares between (or sum of squares for the treaments)
            SS_b = np.sum(size*(mean - mu_all)**2 for size, mean in zip(n, mu))

            # sum of squares within (or sum of squares for the error)
            SS_w = np.sum(np.sum(x - mean)**2 for x, mean in zip(samples, mu))

            # degrees of freedom
            df_b = len(samples) - 1  # degrees of freedom between groups
            df_w = total_n - len(samples)  # degrees of freedom within groups
            
            # mean squares between or mean squares for the treatments
            MS_b = SS_b / df_b

            # mean squares within or mean squares for the error
            MS_w = SS_w / df_w

            # F-statistic, p_value and critical values calculation
            F_stat = MS_b / MS_w
            p_value = 1 - stats.f.cdf(F_stat, df_b, df_w)
            F_crit = stats.f.ppf(1 - alpha, df_b, df_w)

            # effect size (Eta-squared)
            eta_square = SS_b /(SS_b + SS_w)

            # conclusion
            if p_value < alpha:
                conclusion = f"Reject H₀: μ₁ = μ₂ = ... μ_k and Accept H₁: At least one μ_i ≠ μ_j"
            else:
                conclusion = f"Fail to Reject H₀: μ₁ = μ₂ = ... μ_k and Reject H₁: At least one μ_i ≠ μ_j"

            # visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # box plots
            ax1.boxplot(samples)
            ax1.set_xlabel('Groups')
            ax1.set_ylabel('Values')
            ax1.set_title('Box Plots of Samples')

            # F-distribution
            x = np.linspace(0, stats.f.ppf(0.999, df_b, df_w), 1000)
            f_pdf = stats.f.pdf(x, df_b, df_w)
            ax2.plot(x, f_pdf, 'b-', label='F-distribution')
            ax2.axvline(F_stat, color='r', linestyle='--', label='F-statistic')
            ax2.axvline(F_crit, color='g', linestyle='--', label='Critical Value (F_crit)')

            # Shade rejection region
            ax2.fill_between(x[x >= F_crit], f_pdf[x >= F_crit], color='r', alpha=0.3)
            
            ax2.set_title('F-Distribution with Test Statistic')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

            # Post-hoc tests with Tukey HSD
            if p_value < alpha:
                print("\n=====> Post-hoc Analysis (Tukey HSD) <=====")
                all_data = np.concatenate(samples)
                groups = np.concatenate([[f"Group {i}"] * len(sample) for i, sample in enumerate(samples)])
                tukey_results = pairwise_tukeyhsd(endog=all_data, groups=groups, alpha=alpha)
                print(tukey_results)

            results ={
                'test_type': 'One-Way ANOVA',
                'F_statistic': F_stat,
                'degrees_of_freedom': (df_b, df_w),
                'p_value': p_value,
                'critical_value': F_crit,
                'sample_sizes': n,
                'effect_size':{
                    'eta_square': eta_square,
                    'interpretation': ('Small' if eta_square < 0.06 else 'Medium' if eta_square < 0.14 else 'Large')
                },
                'confidence_level': 1 - alpha,
                'assumptions':{
                    'normality': normalities,
                    'homoscedasticity': homoscedasticity
                },
                'interpretation': conclusion,
            }
        # in case of data is NOT normally distributed or NOT homoscedastic, proceed with Welch's test
        elif normal_distribution and not (homoscedastic_bartlett or homoscedastic_levene):
            welch_stat, welch_p, degfree, ci_interval = stats.ttest_ind(*samples, equal_var=False, alternative='two-sided')
            if welch_p < alpha:
                print("\n=====> Post-hoc Analysis (Tukey HSD) <=====")
                all_data = np.concatenate(samples)
                groups = np.concatenate([[f"Group {i}"] * len(sample) for i, sample in enumerate(samples)])
                tukey_results = pairwise_tukeyhsd(endog=all_data, groups=groups, alpha=alpha)
                print(tukey_results)
                conclusion = f"Reject H₀: μ₁ = μ₂ = ... μ_k and Accept H₁: At least one μ_i ≠ μ_j"
            else:
                conclusion = f"Fail to Reject H₀: μ₁ = μ₂ = ... μ_k and Reject H₁: At least one μ_i ≠ μ_j"
            
            results ={
                'test_type': 'Welch ANOVA',
                'Test statistic': welch_stat,
                'p_value': welch_p,
                'degrees_of_freedom': degfree,
                'confidence_interval': ci_interval,
                'normality':normalities,
                'homoscedasticity':homoscedasticity,
                'interpretation':conclusion
            }
        elif not normal_distribution: # if not normal, condemn the test and suggests Kruskal-Wallis
            print("""This test has no statistical relevance due to assumptions of normality and equality of variances being violated
                Consider using Kruskal-Wallis H-test (non-parametric alternative)
                or transforming your data (e.g. log transformation)
                """)
            results = {
                'test_type': 'ANOVA',
                'statistical_relevance': False,
                'interpretation': "Assumptions of normality and equality of variances are violated. Consider using Kruskal-Wallis H-test."
            }
    elif locals().get('homoscedastic_levene', False) == False:
        if (normal_distribution and homoscedastic_bartlett) == True:
            # basic statistics
            n = [len(sample) for sample in samples] # vector of samples sizes
            total_n = np.sum(n)
            mu = [np.mean(sample) for sample in samples] # vector for samples means
            var = [np.var(sample) for sample in samples] # vector of samples variances
            
            # overall mean
            mu_all = np.mean([mean*size for mean, size in zip(mu, n)]) / total_n

            # sum of squares between (or sum of squares for the treaments)
            SS_b = np.sum(size*(mean - mu_all)**2 for size, mean in zip(n, mu))

            # sum of squares within (or sum of squares for the error)
            SS_w = np.sum(np.sum(x - mean)**2 for x, mean in zip(samples, mu))

            # degrees of freedom
            df_b = len(samples) - 1  # degrees of freedom between groups
            df_w = total_n - len(samples)  # degrees of freedom within groups
            
            # mean squares between or mean squares for the treatments
            MS_b = SS_b / df_b

            # mean squares within or mean squares for the error
            MS_w = SS_w / df_w

            # F-statistic, p_value and critical values calculation
            F_stat = MS_b / MS_w
            p_value = 1 - stats.f.cdf(F_stat, df_b, df_w)
            F_crit = stats.f.ppf(1 - alpha, df_b, df_w)

            # effect size (Eta-squared)
            eta_square = SS_b /(SS_b + SS_w)

            # conclusion
            if p_value < alpha:
                conclusion = f"Reject H₀: μ₁ = μ₂ = ... μ_k and Accept H₁: At least one μ_i ≠ μ_j"
            else:
                conclusion = f"Fail to Reject H₀: μ₁ = μ₂ = ... μ_k and Reject H₁: At least one μ_i ≠ μ_j"

            # visualizations
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # box plots
            ax1.boxplot(samples)
            ax1.set_xlabel('Groups')
            ax1.set_ylabel('Values')
            ax1.set_title('Box Plots of Samples')

            # F-distribution
            x = np.linspace(0, stats.f.ppf(0.999, df_b, df_w), 1000)
            f_pdf = stats.f.pdf(x, df_b, df_w)
            ax2.plot(x, f_pdf, 'b-', label='F-distribution')
            ax2.axvline(F_stat, color='r', linestyle='--', label='F-statistic')
            ax2.axvline(F_crit, color='g', linestyle='--', label='Critical Value (F_crit)')

            # Shade rejection region
            ax2.fill_between(x[x >= F_crit], f_pdf[x >= F_crit], color='r', alpha=0.3)
            
            ax2.set_title('F-Distribution with Test Statistic')
            ax2.legend()
            
            plt.tight_layout()
            plt.show()

            # Post-hoc tests with Tukey HSD
            if p_value < alpha:
                print("\n=====> Post-hoc Analysis (Tukey HSD) <=====")
                all_data = np.concatenate(samples)
                groups = np.concatenate([[f"Group {i}"] * len(sample) for i, sample in enumerate(samples)])
                tukey_results = pairwise_tukeyhsd(endog=all_data, groups=groups, alpha=alpha)
                print(tukey_results)

            results ={
                'test_type': 'One-Way ANOVA',
                'F_statistic': F_stat,
                'degrees_of_freedom': (df_b, df_w),
                'p_value': p_value,
                'critical_value': F_crit,
                'sample_sizes': n,
                'effect_size':{
                    'eta_square': eta_square,
                    'interpretation': ('Small' if eta_square < 0.06 else 'Medium' if eta_square < 0.14 else 'Large')
                },
                'confidence_level': 1 - alpha,
                'assumptions':{
                    'normality': normalities,
                    'homoscedasticity': homoscedasticity
                },
                'interpretation': conclusion,
            }
        # in case of data is NOT normally distributed or NOT homoscedastic, proceed with Welch's test
        elif normal_distribution and not (homoscedastic_bartlett or homoscedastic_levene):
            welch_stat, welch_p, degfree, ci_interval = stats.ttest_ind(*samples, equal_var=False, alternative='two-sided')
            if welch_p < alpha:
                print("\n=====> Post-hoc Analysis (Tukey HSD) <=====")
                all_data = np.concatenate(samples)
                groups = np.concatenate([[f"Group {i}"] * len(sample) for i, sample in enumerate(samples)])
                tukey_results = pairwise_tukeyhsd(endog=all_data, groups=groups, alpha=alpha)
                print(tukey_results)
                conclusion = f"Reject H₀: μ₁ = μ₂ = ... μ_k and Accept H₁: At least one μ_i ≠ μ_j"
            else:
                conclusion = f"Fail to Reject H₀: μ₁ = μ₂ = ... μ_k and Reject H₁: At least one μ_i ≠ μ_j"
            
            results ={
                'test_type': 'Welch ANOVA',
                'Test statistic': welch_stat,
                'p_value': welch_p,
                'degrees_of_freedom': degfree,
                'confidence_interval': ci_interval,
                'normality':normalities,
                'homoscedasticity':homoscedasticity,
                'interpretation':conclusion
            }
        elif not normal_distribution: # if not normal, condemn the test and suggests Kruskal-Wallis
            print("""This test has no statistical relevance due to assumptions of normality and equality of variances being violated
                Consider using Kruskal-Wallis H-test (non-parametric alternative)
                or transforming your data (e.g. log transformation)
                """)
            results = {
                'test_type': 'ANOVA',
                'statistical_relevance': False,
                'interpretation': "Assumptions of normality and equality of variances are violated. Consider using Kruskal-Wallis H-test."
            }

    return results

# Non-parametric tests
def independent_chisquare_test():# (NOT IMPLEMENTED)
    pass

def independent_exact_fisher():# (NOT IMPLEMENTED)
    pass

def dependent_mcnemar():# (NOT IMPLEMENTED)
    pass

def paired_wilcoxon():# (NOT IMPLEMENTED)
    pass

def independent_mannwhitneyu():# (NOT IMPLEMENTED)
    pass

def kruskal_wallis_test():# (NOT IMPLEMENTED)
    pass

def friedman_test():# (NOT IMPLEMENTED)
    pass

#===========> Sample Size Calculation
def sample_size_calculation(method, confidence_level, margin_error, **kwargs) -> Union[int, Tuple[int, dict]]:
    """
    =============================> Brief Description <===========================
    -----------------------------------------------------------------------------
    Calculate sample size using different statistical methods.

    =============================> PARAMETERS <==================================
    -----------------------------------------------------------------------------   
    >> method : int
        1: Sample size for proportion
        2: Sample size for mean
        3: Sample size for hypothesis testing (two means)
        4: Sample size for hypothesis testing (two proportions)

    >> confidence_level : float
        Desired confidence level (default: 0.95)

    >> margin_error : float
        Desired margin of error (default: 0.05)

    >> **kwargs : dict
        Additional parameters specific to each method

        > Method 1:
            -> proportion : float, default = 0.5 (hypothetical expected proportion)
        > Method 2:
            -> std_dev : float (population or sample std_dev)
        > Method 3:
            -> std_dev : float (pooled std_dev of both populations or samples)
            -> effect_size : float (minimum detectable difference between means)
            -> power : float, default = 0.8 (desired power of the test)
        > Method 4:
            -> p1 : float (proportion of the first population or sample)
            -> p2 : float (proportion of the second population or sample)
            -> power : float, default = 0.8 (desired power of the test)
    =============================> OUTPUT <=====================================
    ----------------------------------------------------------------------------
    >> tuple
        (sample_size, details_dict)
    """
    
    # Input validation
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if not margin_error > 0:
        raise ValueError("Margin of error must be greater than 0")
    if method not in [1, 2, 3, 4]:
        raise ValueError("Method must be 1, 2, 3, or 4")

    # Calculate Z-score for confidence level
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    details = {
        'confidence_level': confidence_level,
        'z_score': z_score,
        'margin_error': margin_error
    }

    try:
        if method == 1:
            # Method 1: Sample size for proportion
            p = kwargs.get('proportion', 0.5)  # Default to 0.5 if not provided
            if not 0 <= p <= 1:
                raise ValueError("Proportion must be between 0 and 1")
            
            n = (z_score**2) * p * (1-p) / (margin_error**2)
            details.update({
                'method': 'Proportion',
                'proportion': p
            })

        elif method == 2:
            # Method 2: Sample size for mean
            if 'std_dev' not in kwargs:
                raise ValueError("Standard deviation (std_dev) is required for method 2")
            std_dev = kwargs['std_dev']
            
            n = (z_score**2) * std_dev**2 / (margin_error**2)
            details.update({
                'method': 'Mean',
                'std_dev': std_dev
            })

        elif method == 3:
            # Method 3: Sample size for hypothesis testing (two means)
            if not all(k in kwargs for k in ['std_dev', 'effect_size', 'power']):
                raise ValueError("std_dev, effect_size, and power are required for method 3")
            if not effect_size > 0:
                raise ValueError("Error! 'effect_size' must not be 0")
            
            std_dev = kwargs['std_dev']
            effect_size = kwargs['effect_size']
            power = kwargs['power']
            
            if not 0 < power < 1:
                raise ValueError("Power must be between 0 and 1")
            
            z_alpha = z_score
            z_beta = stats.norm.ppf(power)
            
            n = 2 * ((z_alpha + z_beta)**2 * std_dev**2) / (effect_size**2)
            details.update({
                'method': 'Two Means',
                'std_dev': std_dev,
                'effect_size': effect_size,
                'power': power,
                'z_beta': z_beta
            })

        elif method == 4:
            # Method 4: Sample size for hypothesis testing (two proportions)
            if not all(k in kwargs for k in ['p1', 'p2', 'power']):
                raise ValueError("p1, p2, and power are required for method 4")
            if p1 == p2:
                raise ValueError("Error! 'p1' and 'p2' must not be equal.")
            
            p1 = kwargs['p1']
            p2 = kwargs['p2']
            power = kwargs['power']
            
            if not (0 <= p1 <= 1 and 0 <= p2 <= 1):
                raise ValueError("Proportions must be between 0 and 1")
            if not 0 < power < 1:
                raise ValueError("Power must be between 0 and 1")
            
            p_avg = (p1 + p2) / 2
            z_alpha = z_score
            z_beta = stats.norm.ppf(power)
            
            numerator = (z_alpha * math.sqrt(2 * p_avg * (1 - p_avg)) + 
                        z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2
            denominator = (p1 - p2)**2
            
            n = numerator / denominator
            details.update({
                'method': 'Two Proportions',
                'p1': p1,
                'p2': p2,
                'power': power,
                'z_beta': z_beta
            })

        # Round up to the nearest whole number
        n = math.ceil(n)
        details['sample_size'] = n
        
        return n, details

    except Exception as e:
        raise ValueError(f"Error in calculation: {str(e)}")

#===========> Regression Analysis
## Linear Regression
# Linear Regression Model
def linear_regression(x, y):# (NOT IMPLEMENTED)
    """
    Generate a linear regression model
    """
    pass

# Multiple Regression Model
def multiple_regression(x, y):# (NOT IMPLEMENTED)
    """
    Generate a multiple regression model
    """
    pass

# Diagnostics
def regression_diagnostics(model):# (NOT IMPLEMENTED)
    """
    Performs complete analysis of residuals, multicolinearity, significant parameters
    """
    pass

# Features selection
def feature_selection(x, y, method):# (NOT IMPLEMENTED)
    """
    Analyses which features contributes the most to the model quality considering numerous indicative
    metrics
    """
    pass

### Time-series Analysis
def time_series_analysis(df, method):# (NOT IMPLEMENTED)
    """
    Perform time-series analysis on the dataset using the specified method.

    >> Decomposition
        - separates the time series into trend, seasonal, and residual components
        - useful for understanding underlying patterns

    >> ARIMA (AutoRegressive Integrated Moving Average)
        - models the time series as a combination of autoregressive and moving average components
        - suitable for forecasting and trend analysis

    >> SARIMA (Seasonal AutoRegressive Integrated Moving Average)
        - extends ARIMA to handle seasonal patterns
        - useful for forecasting with seasonal data

    >> Exponential Smoothing
        - applies weighted averages of past observations to forecast future values
        - useful for smoothing out noise in the data

    >> Seasonal Decomposition of Time Series (STL)
        - decomposes the time series into seasonal, trend, and residual components
        - robust to outliers and non-linear trends
    """
    pass

#===========> Reliability Analysis
# Failure Patterns Analysis
def failure_patterns_analysis(df, method):# (NOT IMPLEMENTED)
    """
    Analyze patterns in equipment failures based on performance metrics such as:

    >> Mean Time Between Failures (MTBF)
    >> Mean Time To Repair (MTTR)
    >> Failure Rate Analysis
    >> Reliability Function Estimation
    >> Hazard Rate Analysis
    """
    pass

# Reliability Metrics Calculation
def reliability_metrics(df, method):# (NOT IMPLEMENTED)
    """
    Calculate key reliability metrics:

    >> Availability
    >> Reliability
    >> Maintainability
    >> OEE
    >> System Reliability
    """
    pass

# Survival Analysis
def survival_analysis(df, method):# (NOT IMPLEMENTED)
    """
    Perform Survival analysis on equipment:

    >> Kaplan-Meier Estimation
    >> Cox Proportional Hazards
    >> Survival Curves
    >> Censored Data Analysis
    """
    pass


    """
    Analyze spare parts inventory:

    >> Demand Forecasting
    >> Stock Level Optimization
    >> Critical Spares Analysis
    >> Lead Time Analysis
    >> ABC Analysis
    """
    pass