import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
import matplotlib.gridspec
import seaborn as sns
from sklearn.decomposition import PCA


def kmeans(dataset, numberOfClusters):
    kmeans = KMeans(n_clusters=numberOfClusters)
    kmeans.fit(dataset)
    y_kmeans = kmeans.predict(dataset)
    return [y_kmeans, kmeans.cluster_centers_]

def kmeans_pp(dataset, numberOfClusters):
    kmeans = KMeans(n_clusters=numberOfClusters, init='k-means++')
    kmeans.fit(dataset)
    y_kmeans = kmeans.predict(dataset)
    return [y_kmeans, kmeans.cluster_centers_]

def agglomerative(dataset, numberOfClusters):
    aggloclust = AgglomerativeClustering(n_clusters=numberOfClusters).fit_predict(dataset)
    return aggloclust

def dbscan(dataset, metric):
    db = DBSCAN(metric=metric).fit_predict(dataset)
    return db

def zad1():
    dataset = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv', usecols=[*range(55, 107)])
    return dataset

def zad2():
    dataset1 = zad1()
    dataset2 = zad1()
    numberOfClusters = int(input('Podaj ilosc klastrow: ') or 5)
    if numberOfClusters < 1:
        numberOfClusters = 5
    kmeans_result = kmeans(dataset1, numberOfClusters)
    kmeans_pp_result = kmeans_pp(dataset2, numberOfClusters)
    dataset1['cluster'] = kmeans_result[0]
    dataset2['cluster'] = kmeans_pp_result[0]
    reduced_data1 = PCA(n_components=2).fit_transform(dataset1)
    results1 = pd.DataFrame(reduced_data1, columns=['pca1', 'pca2'])
    reduced_data2 = PCA(n_components=2).fit_transform(dataset2)
    results2 = pd.DataFrame(reduced_data2, columns=['pca1', 'pca2'])
    reduced_centers1 = PCA(n_components=2).fit_transform(kmeans_result[1])
    results_centers1 = pd.DataFrame(reduced_centers1, columns=['pca1','pca2'])
    reduced_centers2 = PCA(n_components=2).fit_transform(kmeans_pp_result[1])
    results_centers2 = pd.DataFrame(reduced_centers2, columns=['pca1', 'pca2'])
    fig, axs = plt.subplots(2)
    axs[0].set_title("K-Means")
    axs[1].set_title("K-Means++")

    sns.scatterplot(x="pca1", y="pca2", hue=dataset1['cluster'], style=dataset1['cluster'], data=results1, ax=axs[0])
    sns.scatterplot(x="pca1", y="pca2", data=results_centers1, ax=axs[0])
    sns.scatterplot(x="pca1", y="pca2", hue=dataset2['cluster'], style=dataset2['cluster'], data=results2, ax=axs[1])
    sns.scatterplot(x="pca1", y="pca2", data=results_centers2, ax=axs[1])

    plt.show()

def zad3():
    dataset1 = zad1()
    dataset2 = zad1()
    metrics = ['euclidean', 'cosine', 'cityblock']
    distanceMetricNr = int(input("Wybierz numer metryki (domyslnie 0) - 0: euclidean, 1: cosine, 2: cityblock: ") or 0)
    if distanceMetricNr > 2 or distanceMetricNr < 0:
        distanceMetricNr = 0
    distanceMetric = metrics[distanceMetricNr]
    numberOfClusters = int(input('Podaj ilosc klastrow: ') or 5)
    if numberOfClusters < 1:
        numberOfClusters = 5
    dataset1['cluster'] = agglomerative(dataset1, numberOfClusters)
    dataset2['cluster'] = dbscan(dataset2, distanceMetric)
    g = sns.clustermap(dataset1)
    g.gs.update(left=0.05, right=0.45)
    gs2 = matplotlib.gridspec.GridSpec(1, 1, left=0.6)
    ax2 = g.fig.add_subplot(gs2[0])
    reduced_data1 = PCA(n_components=2).fit_transform(dataset2)
    results1 = pd.DataFrame(reduced_data1, columns=['pca1', 'pca2'])
    sns.scatterplot(x="pca1", y="pca2", hue=dataset2['cluster'], style=dataset2['cluster'], data=results1, ax=ax2)

    plt.show()

# zad1()

# zad2()

zad3()







