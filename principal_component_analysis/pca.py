# -*- coding:  utf-8 -*
"""
This module is used to practice PCA algorithm
Authors: shiduo
Date:2016年7月25日
"""
from numpy import *
import matplotlib
import matplotlib.pyplot as plt
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr]
    return mat(datArr)

def pca(dataMat, topNfeat=9999999):
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals #remove mean
    covMat = cov(meanRemoved, rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)            #sort, sort goes smallest to largest
    eigValInd = eigValInd[:-(topNfeat+1):-1]  #cut off unwanted dimensions
    redEigVects = eigVects[:,eigValInd]       #reorganize eig vects largest to smallest
    lowDDataMat = meanRemoved * redEigVects#transform data into new dimensions
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat
if __name__ == '__main__':
    dataMat = loadDataSet('../data/pca.txt')
    lowDDataMat, reconMat = pca(dataMat,1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0], dataMat[:,1], marker='^', s=90)
    ax.scatter(reconMat[:,0], reconMat[:,1], marker='o', s=50, c='red')
    plt.show()
