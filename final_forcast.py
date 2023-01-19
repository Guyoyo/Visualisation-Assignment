import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as opt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score


def load_and_manipulate(data1, data2):
    """
    This function loads two data files and performs data manipulation on them.
    It reads the csv files, drops unnecessary columns, melts the dataframe,
    concatenates the two dataframes, pivots the data, and removes unwanted rows.
    The final dataframe is returned.
    """
    df1 = pd.read_csv(data1, skiprows = 4)
    df1 = df1.drop('Unnamed: 66', axis =1)
    df1 = df1.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value')
    
    df2 = pd.read_csv(data2, skiprows = 4)
    df2 = df2.drop('Unnamed: 66', axis =1)
    df2 = df2.melt(id_vars = ['Country Name',
                                           'Country Code',
                                           'Indicator Name',
                                           'Indicator Code'],
                                var_name = 'year',
                                value_name = 'value')
    
    dataframe = pd.concat([df1, df2])
    dataframe = dataframe[['Country Name', 'Indicator Name', 'year', 'value']].copy()
    
    dataframe = dataframe.pivot(index=['Country Name', 'year'],
                                columns='Indicator Name', 
                                values='value').reset_index()
    dataframe['year'] = dataframe['year'].astype(int)
    
    exclude_list = ['Africa Eastern and Southern','Arab World','Caribbean small states','Central African Republic', 'Central Europe and the Baltics',
        'Early-demographic dividend', 'East Asia & Pacific',
       'East Asia & Pacific (IDA & IBRD countries)',
       'East Asia & Pacific (excluding high income)','Europe & Central Asia',
       'Europe & Central Asia (IDA & IBRD countries)',
       'Europe & Central Asia (excluding high income)', 'European Union',
       'Fragile and conflict affected situations','French Polynesia','Heavily indebted poor countries (HIPC)',
       'High income', 'IBRD only',
       'IDA & IBRD total', 'IDA blend', 'IDA only', 'IDA total','Late-demographic dividend',
       'Latin America & Caribbean',
       'Latin America & Caribbean (excluding high income)',
       'Latin America & the Caribbean (IDA & IBRD countries)',
       'Least developed countries: UN classification', 'Low & middle income', 'Low income', 'Lower middle income',
       'Middle East & North Africa',
       'Middle East & North Africa (IDA & IBRD countries)',
       'Middle East & North Africa',
       'Middle East & North Africa (excluding high income)',
       'Middle income', 'Not classified',
       'OECD members', 'Other small states',
       'Pacific island small states','Post-demographic dividend',
       'Pre-demographic dividend','Small states','South Asia (IDA & IBRD)','Sub-Saharan Africa', 
       'Sub-Saharan Africa (IDA & IBRD countries)',
       'Sub-Saharan Africa (excluding high income)','Upper middle income', 'West Bank and Gaza',
             'World','Africa Western and Central',
               'North America','Euro area', 'South Asia']
    dataframe = dataframe[~dataframe['Country Name'].isin(exclude_list)]

    return dataframe


# load and manipulate data
data = load_and_manipulate('API_EN.ATM.GHGT.KT.CE_DS2_en_csv_v2_4770432.csv', 
                           'API_EG.ELC.ACCS.ZS_DS2_en_csv_v2_4771419.csv')

# display first five rows of data
data.head()

# check for missing values
data.isnull().sum()

# drop missing values
data = data.dropna()

# filter data to only include year greater than 1999
data = data[data['year']>1999].copy()

# create a copy of data for 2019
cluster_data = data[data['year']==2019].copy()

# scale data
scaled_data = MinMaxScaler().fit_transform(cluster_data.drop(['Country Name', 'year'], axis = 1))

# create an empty list to store WCSS values
wcss = []

# loop through the range of 1 to 11 to determine optimal number of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# plot the WCSS values to determine the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

# fit the model to the data
kmeans.fit(scaled_data)

# Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(scaled_data, kmeans.labels_)}')

# predict the clusters
y_pred = kmeans.fit_predict(cluster_data.drop(['Country Name', 'year'], axis = 1))

# add the cluster column to the data
cluster_data['cluster'] = y_pred

# display the first 2 rows of data
cluster_data.head(2)

# create a scatter plot of the data
plt.figure(figsize=(12,6))
sns.set_palette("pastel")  
sns.scatterplot(y= cluster_data['Access to electricity (% of population)'],
                x= cluster_data['Total greenhouse gas emissions (kt of CO2 equivalent)'], 
                hue= cluster_data['cluster'], 
                palette='bright')
plt.title('Total greenhouse gas emissions (kt of CO2 equivalent) (2019)', fontsize = 18)
plt.show()

#create a copy of data for 2000
cluster_2 = data[data['year']==2000].copy()

#display the first 5 rows of data
cluster_2.head()

#scale data
_scaled_data = MinMaxScaler().fit_transform(cluster_2.drop(['Country Name', 'year'], axis = 1))

#create an empty list to store WCSS values
wcss = []

#loop through the range of 1 to 11 to determine optimal number of clusters
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

#plot the WCSS values to determine the optimal number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#initialize the model with the optimal number of clusters
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

#fit the model to the data
kmeans.fit(_scaled_data)

#Use the silhouette score to evaluate the quality of the clusters
print(f'Silhouette Score: {silhouette_score(_scaled_data, kmeans.labels_)}')

#predict the clusters
y_pred_ = kmeans.fit_predict(cluster_2.drop(['Country Name', 'year'], axis = 1))

#add the cluster column to the data
cluster_2['cluster'] = y_pred_

#create a scatter plot of the data
plt.figure(figsize=(12,6))
sns.set_palette("pastel")
sns.scatterplot(y= cluster_2['Access to electricity (% of population)'],
x= cluster_2['Total greenhouse gas emissions (kt of CO2 equivalent)'],
hue= cluster_2['cluster'],
palette='bright')
plt.title('Total greenhouse gas emissions (kt of CO2 equivalent) (2000)', fontsize = 18)
plt.show()

# This code is checking unique values of 'Country Name' for each cluster and printing them out
for i in np.arange(0,2):
    a = cluster_data[cluster_data['cluster']==i]['Country Name'].unique()
    b = cluster_2[cluster_2['cluster']==i]['Country Name'].unique()
    print(a)
    print("")
    print(b)
    print("")

def exponential(t, n0, g):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    
    t = t - 1960.0
    f = n0 * np.exp(g*t)
    
    return f

def expfunc(x, y, z, s):
    """Calculates the expfunc function with parameters x, y, z and s."""
    return y * np.exp(-z * x) + s

def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f

# Error ranges calculation
def err_ranges(x, func, param, sigma):
    """
    This function calculates the upper and lower limits of function, parameters and
    sigmas for a single value or array x. The function values are calculated for 
    all combinations of +/- sigma and the minimum and maximum are determined.
    This can be used for all number of parameters and sigmas >=1.
    """

    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    # Create a list of tuples of upper and lower limits for parameters
    uplow = []   
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    # calculate the upper and lower limits
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper

def fit_and_forcast(function, data, country, p0):
    """
    Fits the given function to the data, plots the fit and forecast.
    Also plots the confidence interval of the forecast.
    """
    if function == logistic:
        param, covar = opt.curve_fit(function, data["year"], data["Total greenhouse gas emissions (kt of CO2 equivalent)"],
                             p0= p0)
    else:
        param, covar = opt.curve_fit(exponential, data["year"], data["Total greenhouse gas emissions (kt of CO2 equivalent)"],
                             p0= p0)
        
    data["fit"] = function(data["year"], *param)

    data.plot("year", ["Total greenhouse gas emissions (kt of CO2 equivalent)", "fit"], figsize=(12, 8) )
    plt.title( country + " Total Greenhouse Gas Emission Model Fit", fontsize = 16)
    plt.show()
    
    year = np.arange(data.year.min(), 2031)
    forecast = function(year, *param)

    sigma = np.sqrt(np.diag(covar))
    low, up = err_ranges(year, function, param, sigma)

    data.plot("year", ["Total greenhouse gas emissions (kt of CO2 equivalent)"], figsize=(12, 8))
    plt.plot(year, forecast, label="forecast")
    plt.title(country + " Total Greenhouse Gas Emission Forcast", fontsize = 16)

    data.plot("year", ["Total greenhouse gas emissions (kt of CO2 equivalent)"],  figsize=(12, 8))
    plt.plot(year, forecast, label="forecast")

    plt.fill_between(year, low, up, color="yellow", alpha=0.7)
    plt.xlabel("year")
    plt.ylabel("Total greenhouse gas emissions (kt of CO2 equivalent)")
    plt.title(country + " Total Greenhouse Gas Emission Forcast and Confidence Interval", fontsize = 16)
    plt.legend()
    plt.show()

# Filter data for China and plot emissions over time
china = data[data['Country Name']=='China'].copy()

plt.figure(figsize=(12,6))
china.plot( 'year', "Total greenhouse gas emissions (kt of CO2 equivalent)", figsize=(12, 8) )
plt.title('Total greenhouse gas emissions (kt of CO2 equivalent) in China', fontsize = 16)
plt.show()

# Fit and forecast China's emissions using logistic function
fit_and_forcast(logistic, china, 'China', (3e12, 0.03, 2000.0))

# Filter data for United States and plot emissions over time
us = data[data['Country Name']=='United States'].copy()
plt.figure(figsize=(12,6))
us.plot( 'year', "Total greenhouse gas emissions (kt of CO2 equivalent)",  figsize=(12, 8) )
plt.title('Total greenhouse gas emissions (kt of CO2 equivalent) in USA', fontsize = 16)
plt.show()

# Fit and forecast US emissions using exponential function
fit_and_forcast(exponential, us, 'United States', (1e2, 0.03))

# Filter data for Ghana and plot emissions over time
ghana = data[data['Country Name']=='Ghana'].copy()
plt.figure(figsize=(12,6))
ghana.plot( 'year', "Total greenhouse gas emissions (kt of CO2 equivalent)",  figsize=(12, 8))
plt.title('Total greenhouse gas emissions (kt of CO2 equivalent) in Ghana', fontsize = 16)
plt.show()

# Fit and forecast Ghana's emissions using exponential function
fit_and_forcast(exponential, ghana, 'Ghana', (1e2, 0.03))
