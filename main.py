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

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def zad1():
    dataset = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', header=True)
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


def zad2(numberOfClusters = 5):
    # I’ll use the StandardScaler class . This class implements a type of feature scaling
    # called standardization.Standardization scales, or shifts, the values for each numerical
    # feature in your dataset so that the features have a mean of 0 and standard deviation of 1

    df = zad1()
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

zad2()