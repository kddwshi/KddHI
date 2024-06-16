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

class ReformulatedShapleyRegressionModel():
    def __init__(self, df, model1, model2, M, num_features, data , data_val, load, dataset):
        self.explainer_m = MedianShapleyRegressionModel(df, model1, M, num_features, data , data_val, load, dataset)
        self.time_train=0
        self.name='R-ShapleyRegression'
        self.list_time=[]
        self.list_l2_err=[]
        self.list_ul2_err=[]
        self.list_l2=[]
        self.list_ul2=[]
        self.list_udistance=[]
        self.list_distance=[]
        self.list_usv=[]
        self.list_sv=[]
        self.list_sv_err=[]
        self.list_sv_mean=[]
        self.list_usv_err=[]
        self.list_usv_mean=[]
        self.num_features = num_features

    def compute(self, IDX, x, y, kernelshap_iters, diff_u):
        sve_m1, sve_m2 = self.explainer_m.compute(IDX, x, y, kernelshap_iters)
        ISV1=ref_phi(sve_m1, x, y, self.num_features, diff_u)
        ISV2=ref_phi(sve_m2, x, y, self.num_features, diff_u)
        return [ISV1, ISV2]

class ImprovedShapleyRegressionModel():
    def __init__(self, df, model1, model2, M, num_features, data , data_val, load, dataset):
        self.explainer_m = MedianShapleyRegressionModel(df, model1, M, num_features, data , data_val, load, dataset)
        self.explainer_u = UncertainShapleyRegressionModel(df, model2, M, num_features, data , data_val, load, dataset)
        self.time_train=0
        self.name='I-ShapleyRegression'
        self.list_time=[]
        self.list_l2_err=[]
        self.list_ul2_err=[]
        self.list_l2=[]
        self.list_ul2=[]
        self.list_distance=[]
        self.list_udistance=[]
        self.list_usv=[]
        self.list_sv=[]
        self.list_sv_err=[]
        self.list_sv_mean=[]
        self.list_usv_err=[]
        self.list_usv_mean=[]
        self.num_features = num_features

    def compute(self, IDX, x, y, kernelshap_iters, diff_u):
        sve_m1, sve_m2 = self.explainer_m.compute(IDX, x, y, kernelshap_iters)
        sve_u1, sve_u2 = self.explainer_u.compute(IDX, x, y, kernelshap_iters)

        arr_phi_u1=np.abs(sve_u1[:,y])
        width_phi_u1 = arr_phi_u1*2
        ISV1=ref_phi_U(sve_m1, x, y, width_phi_u1, diff_u)

        arr_phi_u2=np.abs(sve_u2[:,y])
        width_phi_u2 = arr_phi_u2*2
        ISV2=ref_phi_U(sve_m2, x, y, width_phi_u2, diff_u)
        return [ISV1, ISV2]

    
class MedianShapleyRegressionModel(): #THRESHOLD CURRENTLY HARD-CODED
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.model_lam = lambda x: model.predict(x)
        self.marginal_extension = MarginalExtension(data[:20], self.model_lam)
        self.name='M-ShapleyRegression'

    def compute(self, IDX, x, y, kernelshap_iters):
        game = PredictionGame(self.marginal_extension, x, y)
        shap_values, all_results = ShapleyRegression(game, batch_size=32, paired_sampling=True, detect_convergence=True, thresh=0.1, bar=False, return_all=True)
        return shap_values.values, all_results['values'][list(all_results['iters']).index(kernelshap_iters)]

class UncertainShapleyRegressionModel(): #THRESHOLD CURRENTLY HARD-CODED
    def __init__(self, df, model, M, num_features, data , data_val, load, dataset):
        self.model_lam = lambda x: model.predict(x)
        self.marginal_extension = MarginalExtension(data[:8], self.model_lam)
        self.name='U-ShapleyRegression'

    def compute(self, IDX, x, y, kernelshap_iters):
        game = PredictionGame(self.marginal_extension, x, y)
        shap_values, all_results = ShapleyRegression_U(game, batch_size=32, paired_sampling=False, detect_convergence=True, thresh=0.1, bar=False, return_all=True)
        return shap_values.values, all_results['values'][list(all_results['iters']).index(kernelshap_iters)]