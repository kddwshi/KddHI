import numpy as np
import shap
import pandas as pd
import os
# print(os.getcwd())

class Monks():
    def __init__(self):
        self.dataset = "Monks"
        data = np.loadtxt(f'{os.getcwd()}/isv/datasets/data/monks-1.test', dtype=object, delimiter=' ') #SWAP
        self.X = data[:,2:-1]
        self.Y =data[:,1]
        self.X = self.X.astype(np.int16)
        self.Y = self.Y.astype(np.int16)

        data_test = np.loadtxt(f'{os.getcwd()}/isv/datasets/data/monks-1.train', dtype=object, delimiter=' ') #SWAP
        self.X_test = data_test[:,2:-1]
        self.Y_test = data_test[:,1]
        self.X_test = self.X_test.astype(np.int16)
        self.Y_test = self.Y_test.astype(np.int16)
        self.feature_names = ["a1","a2","a3","a4","a5","a6"]


    def get_data(self):
        return self.X, self.Y, self.X_test, self.Y_test, self.feature_names, self.dataset
    

class Wbcd():
    def __init__(self):
        self.dataset="WBCD"
        self.data = pd.read_csv(f'{os.getcwd()}/isv/datasets/data/breast-cancer-wisconsin.data', dtype=object, delimiter=',',header=None,na_values="?")
        self.data.columns=["ID","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class"]
        self.data=self.data.dropna(axis=0)
        self.Y=self.data["Class"]
        self.Y=self.Y.astype(np.int64)
        tmp=[]
        for el in self.Y:
            if el==2:
                tmp.append(1)
            else:
                tmp.append(0)
        self.Y=np.array(tmp)
        self.X=self.data.drop(columns=["ID","Class"]).to_numpy()
        self.feature_names = ["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses"]
        
    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset
    
class Diabetes():
    def __init__(self):
        self.dataset="Diabetes"
        self.data=np.loadtxt(f'{os.getcwd()}/isv/datasets/data/diabetes_data.txt', dtype=object, delimiter=',')
        self.X = self.data[:,:-1]
        self.Y = self.data[:,-1]
        self.mapper={"tested_negative": 0, "tested_positive": 1}
        self.Y =np.array([self.mapper[el] for el in self.Y])
        self.feature_names=["preg","plas","pres","skin","insu","mass","pedi", "age"]

    def get_data(self):
        return self.X, self.Y,None, None, self.feature_names, self.dataset

class Bank():
    def __init__(self):
        self.dataset="Bank"
        self.data=np.loadtxt(f'{os.getcwd()}/isv/datasets/data/data_banknote_authentication.txt', dtype=object, delimiter=',')
        self.X=self.data[:,0:-1]
        self.Y=self.data[:,-1]
        self.feature_names=["variance","skewness","curtosis","entropy"]
        self.Y = self.Y.astype(np.int16)
        self.X = self.X.astype(np.float32)

    def get_data(self):
        return self.X, self.Y, None, None, self.feature_names, self.dataset