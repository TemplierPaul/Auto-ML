import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from os import listdir
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class DataSet():

    def __init__(self):
        self.raw_data = {'train': None, 'validation': None, 'test': None}
        self.data = {'train': None, 'validation': None, 'test': None}
        self.plot_data = {'train': None, 'validation': None, 'test': None}
        self.correlation = None
        self.confusion = None

    def __str__(self):
        return 'Columns: ' + str(self.data['train'].columns.to_list()) + '\nSize: ' + str(len(self.data))

    def importData(self, source, keep_source=True):

        def fetchCSV(file):
            """
            Function importing a csv file
            :param file: String
                    name of the csv file
            :return: pandas.Dataframe with the content of the file
            """
            try:
                # First, determines what separator is used in the csv file : "," or ";"
                sep = ','
                df = pd.read_csv(file, sep=sep, nrows=5,
                                 low_memory=False)  # Imports the first 5 rows to check the format
                if ';' in df.columns[0]:  # Checks if the separator is ',' or ';'
                    sep = ';'
                    df = pd.read_csv(file, sep=sep, nrows=5,
                                     low_memory=False)  # Imports the csv file with the ';' separator if needed
                    if len(df.columns) == 1:  # Raises warning if the file seems to have a different separator
                        raise Warning("Only 1 column in ", file, "\nPlease check separator is ',' or ';'")

                # Checks for the first 2 rows if the first column corresponds to an index
                if df.loc[0].iloc[0] == 0 and df.loc[1].iloc[0] == 1:
                    df = pd.read_csv(file, sep=sep, index_col=0, low_memory=False)  # Imports data with an index column
                else:
                    df = pd.read_csv(file, sep=sep, low_memory=False)  # Imports data with a new idex column
                if keep_source: df['ImportSource'] = file
            except:
                raise Exception('Fetch CSV failed on', file)  # If the previous process failed, raises and Exception
            print('Imported', file)
            return df

        def fetchDir(path):
            """
            Recursive function importing all csv files in a directory
            :param path: String
                    relative path to the directory from where this file is
            :return: list of pandas.Dataframe with the data of a file in each
            """
            files = listdir(path)  # Lists the elements in the directory
            dataframes = []
            for f in files:
                if f[-4:] == '.csv':  # If the file is a .csv, imports it
                    dataframes.append(fetchCSV(path + '/' + f))
                else:  # Else, considers it is a directory and imports its csv files recursively
                    dataframes = dataframes + fetchDir(f)
            return dataframes  # Returns a list of DataFrame

        def fetch(source_fetch):
            """
            Selects the right fetching function depending on the source type
            :param source_fetch: String, List or pandas.Dataframe
            :return: pandas.Dataframe with the whole dataset concatenated
            """
            t = repr(type(source_fetch))
            if t == "<class 'str'>":  # If the source is a String, considers it is a csv file or a directory
                if source_fetch[-4:] == '.csv':  # Imports csv
                    df = fetchCSV(source_fetch)
                else:
                    df = pd.concat(fetchDir(source_fetch),
                                   sort=False)  # Recursively imports all csv files in a directory
            elif t == "<class 'pandas.core.frame.DataFrame'>":  # Uses the source as a DataFrame
                df = source_fetch
                if keep_source: df['ImportSource'] = 'DataFrame'
            elif t == "<class 'list'>":  # Runs recursively on the source to import data
                df = pd.concat([fetch(s) for s in source_fetch])
            else:
                raise Exception('Import Logs - This source type is not supported ', t)
            return df  # Returns a DataFrame

        if self.raw_data is None:
            self.raw_data['train'] = fetch(source)  # Imports data
        else:
            self.raw_data['train'] = pd.concat([self.raw_data['train'], fetch(source)])
        return self

    def clearData(self):
        self.data = {'train': None, 'validation': None, 'test': None}
        self.raw_data = {'train': None, 'validation': None, 'test': None}
        self.plot_data = {'train': None, 'validation': None, 'test': None}
        self.correlation = None
        self.confusion = None
        return self

    def split(self, train=0.6, test=None, shuffle=True):
        if test is None:
            test = (1 - train) / 2
            val = test
        else:
            val = 1 - train - test
        if train < 1:
            self.raw_data['train'], x = train_test_split(self.raw_data['train'], shuffle=shuffle, test_size=test + val)
            self.raw_data['validation'], self.raw_data['test'] = train_test_split(x, shuffle=shuffle,
                                                                                  test_size=test / (test + val))
        return self

    def transform(self): #TODO
        self.data = self.raw_data.copy()
        return self

    def correlationGraph(self, plot=True, annot=True, block=False, verbose=True):
        """
        Computes and displays the correlation graph for the dataset
        :param plot: Boolean
                If True, plots the graph
        :param annot: Boolean
                If True, prints the correlation values on the graph
                Else only plots correlation values with a color scale
                    Useful to keep a clear graph with many features
        :param block: Boolean
                If True, stops the algorithm while the graph is open
        :param verbose: Boolean
                Verbosity
        :return: self
        """
        # Computes the correlation graph
        if self.data is None:
            self.correlation = self.raw_data['train'].corr()
            print("Correlation on raw data")
        else:
            self.correlation = self.data['train'].corr()
            print("Correlation on transformed data")
        if verbose: print(self.correlation)

        # Plots the correlation graph
        if plot:
            plt.figure()
            sns.heatmap(self.correlation,
                        xticklabels=self.correlation.columns.values,
                        yticklabels=self.correlation.columns.values,
                        cmap='coolwarm', annot=annot)
            plt.show(block=block)
        return self

    def plotDataset(self, train_val_test='train', title=None, value=None, discrete=False, block=False, tsne=False,
                    verbose=True):

        if 'val' in train_val_test:
            train_val_test = 'validation'
        if self.data is None:
            raise Exception("No data to plot")

        if title is None:
            title = train_val_test

        if value is None:
            value = pd.DataFrame([0 for _ in range(len(self.data[train_val_test]))])
            value.columns = ['PlotValue']
            discrete = True

        # Projecting from the n-dimensional features space to a 2-dimensional displayable space
        if tsne or (len(self.data[train_val_test].columns) != 2 and self.plot_data[train_val_test] is None):
            # If the datset has more than 60 features, use PCA to reduce it then TSNE
            if len(self.data[train_val_test].columns) > 60:
                if verbose: print('Doing PCA')
                self.plot_data[train_val_test] = pd.DataFrame(PCA(n_components=2).fit_transform(self.data[train_val_test]))
                if verbose: print('PCA done - Doing TSNE')
                self.plot_data[train_val_test] = pd.DataFrame(TSNE(n_components=2).fit_transform(self.plot_data[train_val_test]),
                                              columns=['x_plot', 'y_plot'])
            else:
                if verbose: print('Doing TSNE')
                self.plot_data[train_val_test] = pd.DataFrame(
                    TSNE(n_components=2).fit_transform(self.data[train_val_test]), columns=['x_plot', 'y_plot'])
            if verbose: print('Data projected onto a 2-dimensional space for plotting')
        elif len(self.data[train_val_test].columns) == 2 and self.plot_data[train_val_test] is None:
            self.plot_data = self.data.copy()

        # Initiates a new matplotli.pyplot figure
        plt.figure()
        plt.title(title)
        dataToPlot = []

        value.columns=['PlotValue']

        # For discrete data to plot
        if discrete:
            un = pd.DataFrame(value['PlotValue'].unique())
            un.columns = ['uniqueValues']
            groups = []  # list of possible values for the data to plot (ex: cluster)
            labels = []  # list of Series containing points that have the value at the same position in groups

            # Defines the color map to use
            colormap = sns.color_palette("hls", len(un))

            # Defines the list of possible values and the Series of points with these values
            for index, row in un.iterrows():
                groups.append(self.plot_data[train_val_test][value['PlotValue']==row['uniqueValues']].copy())
                labels.append(row['uniqueValues'])

            # Plots each Series with a different color
            for i in range(len(groups)):
                dataToPlot.append(
                    plt.scatter(groups[i]['x_plot'], groups[i]['y_plot'], s=20, edgecolor='k', c=[colormap[i]],
                                label=labels[i]))
            plt.legend()

        # For continuous data (ex: anomaly score)
        else:
            dataToPlot.append(
                plt.scatter(self.plot_data[train_val_test]['x_plot'], self.plot_data[train_val_test]['y_plot'], s=30,
                            edgecolor='k', c=value['PlotValue']))

            # Prints the color bar on the right side
            plt.colorbar()

        plt.axis('tight')
        plt.show(block=block)
        return self
