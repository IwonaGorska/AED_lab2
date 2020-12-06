from sklearn import datasets
from sklearn import model_selection
import pandas as pd

import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

import tarfile
import urllib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def zad1():
    # normalized = []
    #Normalized 51 - create strings like that in a loop, push to array and use in 'usecols' maybe
#usecols=[55, 107]
    dataset = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', usecols=[*range(55, 107)])
    print(dataset)
    pd.DataFrame = dataset
    valueCount = pd.DataFrame.size
    uniqueValueCount = pd.DataFrame.nunique()
    meanValue = pd.DataFrame.mean()
    numberOfNulls = pd.DataFrame.isna().sum()
    maxValue = pd.DataFrame.max()
    minValue = pd.DataFrame.min()
    commonValue = pd.DataFrame.mode()
    # print("Ilosc wartosci: ", valueCount)
    # print("--------------------")
    # print("Ilosc wartosci unikatowych: ", uniqueValueCount)
    # print("--------------------")
    # print("Wartosc srednia w zbiorze: ", meanValue)
    # print("--------------------")
    # print("Ilosc wartosci null: ", numberOfNulls)
    # print("--------------------")
    # print("Wartosc maksymalna: ", maxValue)
    # print("--------------------")
    # print("Wartosc minimalna: ", minValue)
    # print("--------------------")
    # print("Wartosc najczesciej wystepujaca w zbiorze: ", commonValue)
    # print("********************")
    return dataset


def kmeans(numberOfClusters, palette, markers, df):
    # X, y_true = make_blobs(n_samples=1000, centers=numberOfClusters, cluster_std=0.99, random_state=3042019)
    # df = pd.DataFrame(, columns=['f1', 'f2'])
    # df.head()

    print('Zbiór przed klasteryzacją: \n', df)

    kmeans = KMeans(n_clusters=numberOfClusters)
    kmeans.fit(df)
    y_kmeans = kmeans.predict(df)

    df['sklearn_cluster'] = y_kmeans
    sklearn_centers = kmeans.cluster_centers_

    sns.lmplot(data=df,  fit_reg=False, hue='sklearn_cluster', markers=markers,
               palette=palette).set(title='Wizualizacja grup - KMeans')
    plt.scatter(sklearn_centers[:, 0], sklearn_centers[:, 1], c='black', s=100, alpha=0.5)
    plt.show()

    print('Zbiór po klasteryzacji (kolumna sklearn_cluster to przypisana etykieta klas): \n', df)


def kmeans_pp(numberOfClusters, palette, markers, df):
    # X, y_true = make_blobs(n_samples=1000, centers=numberOfClusters, cluster_std=0.99, random_state=3042019)
    # df = pd.DataFrame(X, columns=['f1', 'f2'])
    # df.head()

    kmeans = KMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(df)
    y_kmeans = kmeans.predict(df)

    df['sklearn_cluster'] = y_kmeans
    sklearn_centers = kmeans.cluster_centers_

    sns.lmplot(data=df,  fit_reg=False, hue='sklearn_cluster', markers=markers,
               palette=palette).set(title='Wizualizacja grup - KMeans++')
    plt.scatter(sklearn_centers[:, 0], sklearn_centers[:, 1], c='black', s=100, alpha=0.5)
    plt.show()


def zad2():

    wholePalette = ['#10fbbb', '#fbe610', '#eb6c6a', '#6aeb6c', '#6c6aeb', '#8b0000', '#002e53', '#fb109e']
    allMarkers = ['o', '*', '+', '^', 'x', 's', 'v', 'w']
    numberOfClusters = int(input("Podaj liczbe klastrow (max 8) lub wciśnij Enter, by przyjąć domyślną wartość (5): ") or 5)
    if numberOfClusters > 8 or numberOfClusters < 1:
        numberOfClusters = 8
    print(numberOfClusters)
    palette = wholePalette[0:numberOfClusters]
    markers = allMarkers[0:numberOfClusters]
    print(palette)
    print(markers)

    df = zad1() # - in the future put this as a parameter into below functions as well
    kmeans(numberOfClusters, palette, markers, df)
    kmeans_pp(numberOfClusters, palette, markers, df)

    # I’ll use the StandardScaler class . This class implements a type of feature scaling
    # called standardization.Standardization scales, or shifts, the values for each numerical
    # feature in your dataset so that the features have a mean of 0 and standard deviation of 1

    # df = zad1()
    # DF <- data.frame

    # dataset, true_labels = make_blobs(
    #     n_samples=200,
    #     centers=3,
    #     cluster_std=2.75,
    #     random_state=42
    # )
    #
    # scaler = StandardScaler()
    # normalizedDataset = scaler.fit_transform(dataset)
    # kmeans = KMeans(
    #     init="random",
    #     n_clusters=numberOfClusters,
    #     n_init=10,
    #     max_iter=300,
    #     random_state=42
    # )
    # kmeans.fit(dataset)
    # # The lowest SSE value
    # print('The lowest SSE value: ', kmeans.inertia_)
    # # Final locations of the centroid
    # print('Final locations of the centroid: \n', kmeans.cluster_centers_)
    # # The number of iterations required to converge
    # print('The number of iterations required to converge: ', kmeans.n_iter_)
    # # The cluster assignments are stored as a one-dimensional NumPy array
    # # in kmeans.labels_. Here’s a look at the first five predicted labels:
    # print('First five predicted labels of cluster assignments: ', kmeans.labels_[:5])

    # kmeans = KMeans(n_clusters=numberOfClusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # pred_y = kmeans.fit_predict(X)
    # plt.scatter(X[:, 0], X[:, 1])
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
    # plt.show()


def agglomerativeClustering(metric, numberOfClusters):
    np.random.seed(1)
    x, _ = make_blobs(n_samples=300, centers=numberOfClusters, cluster_std=.8)

    # x = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', usecols=[*range(55, 107)])

    aggloclust = AgglomerativeClustering(n_clusters=numberOfClusters).fit(x)
    print(aggloclust)

    AgglomerativeClustering(affinity=metric, compute_full_tree='auto',
                            connectivity=None, linkage='ward', memory=None, n_clusters=numberOfClusters)

    labels = aggloclust.labels_

    sns.clustermap(x, metric=metric, standard_scale=1, method="single")

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    # plt.scatter(x[:, 0], x[:, 1], c=[])
    plt.show()



def zad3():
    metricks = ['euclidean', 'cosine', 'cityblock']
    distanceMetricNr = int(input("Wybierz numer metryki (domyslnie 0) - 0: euclidean, 1: cosine, 2: cityblock: ") or 0)
    if distanceMetricNr > 2 or distanceMetricNr < 0:
        distanceMetricNr = 0
    distanceMetric = metricks[distanceMetricNr]
    print(distanceMetric)

    # Additionally user can change also number of clusters
    numberOfClusters = int(
        input("Podaj liczbe klastrow lub wciśnij Enter, by przyjąć domyślną wartość (5): ") or 5)
    if numberOfClusters < 1:
        numberOfClusters = 5
    print(numberOfClusters)

    agglomerativeClustering(distanceMetric, numberOfClusters)

zad3()