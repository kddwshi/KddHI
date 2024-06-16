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


class MonteCarloModel():
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.df = df
        self.model = model
        self.M = M
        self.num_features = num_features
        self.time_train=0
        self.name='R-MC'
        self.list_time=[]
        self.list_l2_err=[]
        self.list_l2=[]
        self.list_ul2_err=[]
        self.list_ul2=[]
        self.list_distance=[]
        self.list_udistance=[]
        self.list_sv=[]
        self.list_usv=[]
        self.list_sv_err=[]
        self.list_usv_err=[]
        self.list_usv_mean=[]
        self.list_sv_mean=[]


    def compute(self, IDX, x, y, kernelshap_iters, diff_u):
        mc_m, mc_u=MonteCarlo_UNI(IDX, self.df, self.model, self.M)
        # print(mc_m)
        ISV1 = ref_phi(mc_m, x, y, self.num_features, diff_u)
        arr_phi_u=np.abs(mc_u[:,y])
        width_phi_u = arr_phi_u*2
        ISV2 = ref_phi_U(mc_m, x, y, width_phi_u, diff_u)
        return [ISV1, ISV2]