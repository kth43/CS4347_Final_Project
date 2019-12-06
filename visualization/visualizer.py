import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
dataFile = os.getcwd()+'/data/creditcard.csv'
allData = pd.read_csv(dataFile)


fraud = allData['Class']==1
noFraud = allData['Class']==0
def plotGraph(feature, show=True, save=False):
    print("Plotting " + feature)
    allData[fraud][feature].value_counts().sort_index().plot(kind='bar')
    plt.title("Fraud " + feature)
    if show:
        plt.show()
    if save:
        plt.savefig(os.getcwd() + '/visualization/' + feature + ' Fraud.png')
    allData[noFraud][feature].value_counts().sort_index().plot(kind='bar')
    plt.title("No Fraud " + feature)
    if show:
        plt.show()
    if save:
        plt.savefig(os.getcwd() + '/visualization/' + feature + ' No Fraud.png')

def binMake(data, numBins):
    #print(numBins)

    ret = np.zeros(numBins)
    ran = math.fabs(allData[data].max())+math.fabs(allData[data].min())

    step = ran/numBins;
    curStep = allData[data].min()

    for i in range(numBins):
        ret[i] = curStep;
        curStep+=step;
    return ret

def visualHelper(data, bin):
    print(data)
    allData[data + ' Bin'] = pd.cut(allData[data], vBins[bin])
    plotGraph(data + ' Bin', show=False)

numBins = 25;
rows, cols = (28, numBins)

vBins = [[0]*cols]*rows
for i in range(1, 29):
    passVal = "V{}".format(i)
    x = binMake(passVal, numBins)
    vBins[i-1]=x;
    visualHelper(passVal, i-1)
    plotGraph(passVal, show=False, save=False)