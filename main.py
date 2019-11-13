from Clustering import *
from Classification import *
from Regression import *

cl = Clusterer().importData('iris.csv', keep_source=False).split(train=0.6).transform()

cl.plotDataset(train_val_test='train')
cl.correlationGraph(block=False)
cl.clustering(n=3)
cl.plotClusters()