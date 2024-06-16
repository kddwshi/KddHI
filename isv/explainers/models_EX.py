from .MonteCarlo.mc import MonteCarlo_UNI
from .ShapReg.shapreg import ShapleyRegression, PredictionGame, MarginalExtension
from .ShapReg.shapreg_LIKE import ShapleyRegression_U
from .myshap.shap.explainers._exact import ExactExplainerU
import shap
# from .shapley_regression import removal, s

from ..evaluation.utils import ref_phi, ref_phi_U


import shap
import torch 
import torch.nn as nn
import time
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReformulatedExactExplainerModel():
    def __init__(self, df, model1, model2, M, num_features, data , data_val, load, dataset):
        self.explainer_m = MedianExactExplainerModel(df, model1, M, num_features, data , data_val, load, dataset)
        self.time_train=0
        self.name='R-EXT'
        self.list_time=[]
        self.list_l2_err=[]
        self.list_l2=[]
        self.list_distance=[]
        self.list_sv=[]
        self.list_sv_mean=[]
        self.list_sv_err=[]
        self.list_distance_versus=[]
        self.num_features = num_features

    def compute(self, IDX, x, y, kernelshap_iters, diff_u):
        sve_m = self.explainer_m.compute(IDX, x, y, kernelshap_iters)
        ISV=ref_phi(sve_m, x, y, self.num_features, diff_u)
        return [ISV]

class ImprovedExactExplainerModel():
    def __init__(self, df, model1, model2, M, num_features, data , data_val, load, dataset):
        self.explainer_m = MedianExactExplainerModel(df, model1, M, num_features, data , data_val, load, dataset)
        self.explainer_u = UncertainExactExplainerModel(df, model2, M, num_features, data , data_val, load, dataset)
        self.time_train=0
        self.name='I-EXT'
        self.list_time=[]
        self.list_l2_err=[]
        self.list_l2=[]
        self.list_distance=[]
        self.list_sv=[]
        self.list_sv_mean=[]
        self.list_sv_err=[]

    def compute(self, IDX, x, y, kernelshap_iters, diff_u):
        sve_m = self.explainer_m.compute(IDX, x, y, kernelshap_iters)
        sve_m = np.array(sve_m)
        sve_u = self.explainer_u.compute(IDX, x, y, kernelshap_iters)
        sve_u = np.array(sve_u)
        arr_phi_u=np.abs(sve_u[:,y])
        width_phi_u = arr_phi_u*2
        ISV=ref_phi_U(sve_m, x, y, width_phi_u, diff_u)
        return [ISV]

    
class MedianExactExplainerModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = shap.explainers.Exact(model.predict, data[:100])
        self.time_train=0
        self.name='M-Exact'

    def compute(self, IDX, x, y, kernelshap_iters):
        sve_m = self.explainer(x)
        sve_m = sve_m.values[0]
        return sve_m

class UncertainExactExplainerModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.explainer = ExactExplainerU(model.predict, data[:100])
        self.time_train=0
        self.name='U-Exact'

    def compute(self, IDX, x, y, kernelshap_iters):
        sve_u = self.explainer(x)
        sve_u = sve_u.values[0]
        return sve_u
