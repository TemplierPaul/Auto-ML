from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN
from Dataset import *


class Clusterer(DataSet):

    def __init__(self, type='kmeans'):
        super(Clusterer, self).__init__()
        self.type = type
        self.model = None
        self.nb_clusters = None
        self.assigned_cluster = {'train': None, 'validation': None, 'test': None}
        self.distance = {'train': None, 'validation': None, 'test': None}
        self.centers = None

    def __str__(self):
        s = 'Clusterer k=' + str(self.nb_clusters)
        if self.centers is None:
            s += ' Not'
        s += " Trained"
        s += '\n' + super(Clusterer, self).__str__()
        return s

    def createModel(self, n=1, init_kmeans='k-means++'):
        """
        Creates the sklearn model depending on self.type
        :param n: Int
                number of clusters
        :param init: {‘k-means++’, ‘random’, or ndarray, or a callable}
                If kmeans is used, selects the algorithm initiation:
                    ‘k-means++’ : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.
                    ‘random’: choose k observations (rows) at random from data for the initial centroids.
                    If an ndarray is passed, it should be of shape (n_clusters, n_features) and gives the initial centers.
                    If a callable is passed, it should take arguments X, k and and a random state and return an initialization.
        :return: sklearn.cluster
        """
        if self.type == "kmeans":
            return KMeans(n_clusters=n, random_state=0, init=init_kmeans)
        if self.type == "miniBatch":
            return MiniBatchKMeans(n_clusters=n, random_state=0, batch_size=6)
        if self.type == "dbscan":
            return DBSCAN(min_samples=10)

    def clustering(self, n=None, centers=None, train_val_test='train'):
        if self.data is None:
            raise Exception('No data')
        if centers is None:
            centers = 'k-means++'
        else:
            n = len(centers)
        if not (n is None):
            self.nb_clusters = n
        elif n is None and self.nb_clusters is None:
            raise Exception("missing number of clusters")

        print("\nClustering model : ", self.type)
        self.model = self.createModel(self.nb_clusters, init_kmeans=centers)  # Create the model
        print(self.model)
        self.assigned_cluster[train_val_test] = pd.DataFrame(
            self.model.fit_predict(self.data[train_val_test]))  # Compute the clusters

        try:
            self.distance[train_val_test] = pd.DataFrame(
                self.model.transform(self.data[train_val_test]).min(1), columns=['Score'])
        except:
            print("no score")
        self.centers = self.model.cluster_centers_
        return self

    def plotClusters(self, train_val_test='train'):
        self.plotDataset(train_val_test=train_val_test, title='Clusters ' + train_val_test,
                         value=self.assigned_cluster[train_val_test], discrete=True, block=False, tsne=False,
                         verbose=True)
        self.plotDataset(train_val_test=train_val_test, title='Distance ' + train_val_test,
                         value=self.distance[train_val_test], discrete=False, block=False, tsne=False, verbose=True)
