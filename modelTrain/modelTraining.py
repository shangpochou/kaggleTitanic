'''
Created on 09102016
@author: ShangpoChou
'''
# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import pandas as pd
import dataAccess.dataReader

def main():
    
    trainingData = dataAccess.dataReader.getTrainingData()

    trainingData.info()

    trainingDataX = trainingData.drop("Survived", axis = 1)
    trainingDataY = trainingData["Survived"]

    #print(trainingDataX)
    
    logreg = LogisticRegression()

    logreg.fit(trainingDataX, trainingDataY)
    
    print(logreg.coef_)
    
    testingData =  dataAccess.dataReader.getTestingData()
    
    #testingDataX = testingData .drop("Survived", axis = 1)
    #testingDataY = testingData ["Survived"]

    print(logreg.score(trainingDataX, trainingDataY))
    
    coeffDf = pd.DataFrame(trainingDataX.columns.delete(0))
    coeffDf.columns = ['Features']
    coeffDf["Coefficient Estimate"] = pd.Series(logreg.coef_[0])

    # preview
    print(coeffDf)

    
if __name__ == "__main__":
    main()