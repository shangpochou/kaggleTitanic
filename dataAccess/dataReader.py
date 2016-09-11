'''
Created on 09102016
@author: ShangpoChou
'''
# Imports

# pandas
import pandas as pd

import numpy as np

import os



os.getcwd() 
os.chdir("C:/Users/ShangpoChou/workplaceVer2/kaggleTitanic/data")

#Testing purpose
def main():
#    getTrainingData()
    df = getTestingData()
    df.info
    
    
# import pylab as P
# titanicDf['Age'].hist()
# P.show()

# plt.hist(titanicDf['Age'])
# plt.show()

def func(row):
    if row["Sex"] == "male":
        return 1
    elif row["Sex"] == "female":
        return 0 
    else:
        return -1

def getDataSet(datapath):
        
    dataDf = pd.read_csv(datapath, header=0)
    dataDf.info()

    dataDf["SexN"] = dataDf.apply(func, axis=1)
    dataDf.info
    dataDf = dataDf.drop(["PassengerId", "Name", "Embarked", "Ticket", "Cabin", "Sex"], axis=1)
    
    dataDf = dataDf.replace([np.inf, -np.inf], np.nan)
    dataDf = dataDf.dropna()
    
    return dataDf


def getTrainingData():
    
    datapath = "./train.csv"
    
    return getDataSet(datapath)

def getTestingData():
    
    datapath = "./test.csv"
    
    return getDataSet(datapath)

    
if __name__ == "__main__":
    main()