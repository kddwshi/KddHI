import shap
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import time
import random
import os
# print("CURRENT WD:",os.getcwd())


# from ..explainers import MonteCarlo, ShapleyRegression, FastSHAP, DASP, DeepExplainer, Exact, MarginalExtension
from ..explainers.models_SR import *
from ..explainers.models_MC import *
from ..explainers.models_EX import *
from .tabulate import tabulate
from .original_model import OriginalModel, OriginalModelVV
from .utils import MedianModel, UncertainModel, get_grand_null_u, MultiTaskModel, MaskLayer1d, KLDivLoss
from .surrogate import Surrogate_VV
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn


class Benchmark():
    def __init__(self, dataset, explainers_reformulated, explainers_improved, metrics, ground_truth_reformulated, ground_truth_improved, num_samples, classifier=None, sample_method='random'):
        self.dataset = dataset
        self.explainers_reformulated = explainers_reformulated
        self.explainers_improved = explainers_improved
        self.explainers_improved_init = {}
        self.explainers_reformulated_init = {}
        self.metrics = metrics
        self.ground_truth_ref=ground_truth_reformulated
        self.ground_truth_imp=ground_truth_improved
        self.ground_truth_name_improved = {}
        self.ground_truth_name_reformulated = {}
        self.num_samples = num_samples
        self.sample_method = sample_method
        self.kernelshap_iters = 128 # default value
        self.classifier = classifier
        self.num_features = {}
        self.surrogate_VV = {}
        

    def run(self, verbose=False, load=False): # CURRENTLY THE GROUND TRUTH IS THE SAME FOR EACH DATASET
        SEED=291297
        for dset_fn in self.dataset:
            dset = dset_fn()
            
            X, Y, X_test, Y_test, feature_names, dataset=dset.get_data()
            if verbose:
                print("-"*100)
                print("Running dataset:", dataset)

            if X_test is None:
                num_features = X.shape[1]
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
                if verbose:
                    print(X.shape, Y.shape, feature_names)
                # data=X_train
                ss = StandardScaler()
                ss.fit(X_train)
                X_train_ss = ss.transform(X_train)
                X_val_ss = ss.transform(X_val)
            else:
                num_features = X.shape[1]
                X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
                if verbose:
                    print(X.shape, Y.shape, X_test.shape, Y_test.shape, feature_names)
                # data=X_train
                ss = StandardScaler()
                ss.fit(X_train)
                X_train_ss = ss.transform(X_train)
                X_val_ss = ss.transform(X_val)
                X_test_ss = ss.transform(X_test)


            if verbose:
                print("\nRunning Black-box Model")

            modelRF = RandomForestClassifier(random_state=SEED)
            modelRF.fit(X_train_ss, Y_train)
            om_VV=OriginalModelVV(modelRF)
            om=OriginalModel(modelRF)
            y_pred=modelRF.predict(X_val_ss)

            if verbose:
                print("Accuracy:", accuracy_score(Y_val, y_pred))


            def original_model_VV(x):
                pred1, pred2 = om_VV.get_pred_VV(x.cpu().numpy()) #MODELLO ORIGINALE, PRED ALWAYS ON POSITIVE CLASS
                return torch.tensor(pred1, dtype=torch.float32, device=x.device), torch.tensor(pred2, dtype=torch.float32, device=x.device)


            device = torch.device('cpu')
            if os.path.isfile(f'{os.getcwd()}/isv/models_LIKE/{dataset}_surrogate_VV.pt'):
                print('\nLoading saved surrogate model')
                surr_VV = torch.load(f'{os.getcwd()}/isv/models_LIKE/{dataset}_surrogate_VV.pt').to(device)
                surrogate_VV = Surrogate_VV(surr_VV, num_features)
            else:
                print('\nTraining surrogate model')
                SEED=291297
                torch.manual_seed(SEED)
                np.random.seed(SEED)
                random.seed(SEED)

                surr_VV=MultiTaskModel(512, num_features).to(device)
                surrogate_VV = Surrogate_VV(surr_VV, num_features)

                surrogate_VV.train_original_model_VV(
                    X_train_ss,
                    X_val_ss,
                    original_model_VV,
                    batch_size=8,
                    max_epochs=200,
                    loss_fn1=KLDivLoss(),
                    loss_fn2=KLDivLoss(),
                    alpha=1,
                    beta=1,
                    validation_samples=10,
                    validation_batch_size=10000,
                    verbose=False,
                    lr=1e-4,
                    min_lr=1e-6,
                    lr_factor=0.5,
                    weight_decay=0.01,
                    debug=False,
                    training_seed=29,
                    lookback=20,
                )

                surr_VV.cpu()
                torch.save(surr_VV, f'{os.getcwd()}/isv/models/{dataset}_surrogate_VV_NEW.pt')
                surr_VV.to(device)

            self.surrogate_VV[dset_fn] = surrogate_VV
            self.num_features[dset_fn] = num_features

            kernelshap_iters = 128
            TH=0.1
            df=pd.DataFrame(X_train_ss, columns=feature_names)
            
            link=nn.Softmax(dim=-1)
            mmodel=MedianModel(surrogate_VV, link, device, num_features)
            umodel=UncertainModel(surrogate_VV, link, device, num_features)

            # initialize reformulated grand truth model
            if self.ground_truth_ref!=None:
                ground_model_ref=self.ground_truth_ref(df, mmodel, umodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                print("\nGround Truth Reformulated:", ground_model_ref.name)
                self.ground_truth_name_reformulated[dset_fn]=ground_model_ref.name
            
            if self.ground_truth_imp!=None:
                ground_model_imp=self.ground_truth_imp(df, mmodel, umodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                print("\nGround Truth Improved:", ground_model_imp.name)
                self.ground_truth_name_improved[dset_fn]=ground_model_imp.name

            if self.ground_truth_ref!=None:
                # initialize explainers as a dictionary name:explainer
                explainers_ref = {}
                explainers_ref[ground_model_ref.name] = ground_model_ref
                print("\nInitializing Explainers Reformulated")
                for explainer in self.explainers_reformulated:
                    print("\tExplainer:", explainer) #still not initialized
                    if explainer == MonteCarloModel:
                        exp = explainer(df, surrogate_VV, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                    else:
                        exp = explainer(df, mmodel, umodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                    explainers_ref[exp.name]=exp
            
                print("\nExplainers Reformulated:", explainers_ref.keys())
                self.explainers_reformulated_init[dset_fn] = explainers_ref
            
            if self.ground_truth_imp!=None:
                # initialize explainers as a dictionary name:explainer
                explainers_imp = {}
                explainers_imp[ground_model_imp.name] = ground_model_imp
                print("\nInitializing Explainers Improved")
                for explainer in self.explainers_improved:
                    print("\tExplainer:", explainer)
                    if explainer == MonteCarloModel:# and explainer not in self.explainers_reformulated:
                        exp = explainer(df, surrogate_VV, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                    # elif explainer == MonteCarloModel and explainer in self.explainers_reformulated:
                    #     exp = explainers_ref["R-MC"]
                    else:
                        exp = explainer(df, mmodel, umodel, 1000, num_features, X_train_ss, X_val_ss, load, dataset)
                    explainers_imp[exp.name]=exp

                print("\nExplainers Improved:", explainers_imp.keys())
                self.explainers_improved_init[dset_fn] = explainers_imp


            # initialize metrics as a dictionary name:metric
            metrics = {}
            for metric in self.metrics:
                mtr=metric()
                metrics[mtr.name]=mtr
            print("\nMetrics:", metrics.keys())

            DATA=X_train_ss
            LABELS=Y_train
            
            print("\nRunning Experiments")
            for IDX in tqdm(range(len(DATA[:self.num_samples]))):
                sample=DATA[IDX:IDX+1]
                label=int(LABELS[IDX:IDX+1][0])
                x_t = torch.tensor(sample, dtype=torch.float32, device=device)
                grand_u, null_u, diff_u = get_grand_null_u(x_t, num_features, label, surrogate_VV)
                
                # print("-"*100)
                # print("DATA:", sample, label)
                # print(diff_u)

                # reformulated models
                if self.ground_truth_ref!=None:
                    for k, expl in explainers_ref.items():
                        # print(k)
                        time_start = time.time()
                        out=expl.compute(IDX, sample, label, kernelshap_iters, diff_u)
                        # if k==self.ground_truth_name_reformulated[dset_fn]:
                        #     print(k, out)
                        time_end = time.time()
                        expl.list_time.append(time_end-time_start)
                        if len(out)>1:
                            # print("\t",k)
                            expl.list_usv.append(out[0])
                            expl.list_sv.append(out[1])

                            sv=out[0]
                            mean_sv=np.mean(sv, axis=1)
                            err_sv=(mean_sv-sv[:,0])
                            
                            # print(out[0])
                            # print(mean_sv)

                            expl.list_sv_mean.append(mean_sv)
                            expl.list_sv_err.append(err_sv)

                            usv=out[1]
                            mean_usv=np.mean(usv, axis=1)
                            err_usv=(mean_usv-usv[:,0])
                            
                            # print(out[1])
                            # print(mean_usv)

                            expl.list_usv_mean.append(mean_usv)
                            expl.list_usv_err.append(err_usv)
                        else:
                            expl.list_sv.append(out[0])
                            sv=out[0]
                            mean_sv=np.mean(sv, axis=1)
                            # print(mean_sv)
                            err_sv=(mean_sv-sv[:,0])
                            expl.list_sv_mean.append(mean_sv)
                            expl.list_sv_err.append(err_sv)

                    # for k,expl in explainers_ref.items():
                    #     print(k, expl.list_sv_mean)

                    ground_sv=ground_model_ref.list_sv[-1]
                    mean_ground_sv=ground_model_ref.list_sv_mean[-1]#np.mean(ground_sv, axis=1)
                    err_ground_sv=ground_model_ref.list_sv_err[-1]
                    
                    # print("")
                    # print(mean_ground_sv)

                    # print(mean_ground_sv, err_ground_sv)
                    # err_ground_sv=(mean_ground_sv-ground_sv[:,0])
                    for k, mtr in metrics.items():
                        for k, expl in explainers_ref.items():
                            if expl.name != ground_model_ref.name:
                                if 'ShapleyRegression' in expl.name:
                                    sv_comparison=expl.list_usv[-1]
                                    mean_sv_comparison=expl.list_usv_mean[-1]
                                    err_sv_comparison=expl.list_usv_err[-1]
                                    
                                    if mtr.name == 'L2':
                                        res_mean=mtr.compute(mean_sv_comparison, mean_ground_sv)
                                        expl.list_ul2.append(res_mean)
                                        res_err=mtr.compute(err_sv_comparison, err_ground_sv)
                                        expl.list_ul2_err.append(res_err)
                                    if mtr.name == 'EuclideanDistance':
                                        res_distance=mtr.compute(sv_comparison, ground_sv)
                                        expl.list_udistance.append(res_distance)
                                
                                if 'R-MC' in expl.name:
                                    sv_comparison=expl.list_usv[-1]
                                    mean_sv_comparison=expl.list_usv_mean[-1]
                                    err_sv_comparison=expl.list_usv_err[-1]
                                    if mtr.name == 'L2':
                                        # print(mean_sv_comparison)
                                        res_mean=mtr.compute(mean_sv_comparison, mean_ground_sv)
                                        expl.list_ul2.append(res_mean)
                                        res_err=mtr.compute(err_sv_comparison, err_ground_sv)
                                        expl.list_ul2_err.append(res_err)
                                    if mtr.name == 'EuclideanDistance':
                                        res_distance=mtr.compute(sv_comparison, ground_sv)
                                        expl.list_udistance.append(res_distance)
                                
                                sv_comparison=expl.list_sv[IDX]
                                mean_sv_comparison=expl.list_sv_mean[IDX]
                                err_sv_comparison=expl.list_sv_err[IDX]

                                # print(k, mean_sv_comparison)

                                if mtr.name == 'L2':
                                    res_mean=mtr.compute(mean_sv_comparison, mean_ground_sv)
                                    expl.list_l2.append(res_mean)
                                    res_err=mtr.compute(err_sv_comparison, err_ground_sv)
                                    expl.list_l2_err.append(res_err)
                                if mtr.name == 'EuclideanDistance':
                                    res_distance=mtr.compute(sv_comparison, ground_sv)
                                    expl.list_distance.append(res_distance)



                                    
                
                if self.ground_truth_imp!=None:
                    for k, expl in explainers_imp.items():
                        time_start = time.time()
                        out=expl.compute(IDX, sample, label, kernelshap_iters, diff_u)
                        time_end = time.time()
                        expl.list_time.append(time_end-time_start)
                        if len(out)>1:
                            expl.list_usv.append(out[0])
                            expl.list_sv.append(out[1])
                            sv=out[0]
                            mean_sv=np.mean(sv, axis=1)
                            err_sv=(mean_sv-sv[:,0])
                            expl.list_sv_mean.append(mean_sv)
                            expl.list_sv_err.append(err_sv)

                            usv=out[1]
                            mean_usv=np.mean(usv, axis=1)
                            err_usv=(mean_usv-usv[:,0])
                            expl.list_usv_mean.append(mean_usv)
                            expl.list_usv_err.append(err_usv)
                        else:
                            expl.list_sv.append(out[0])
                            sv=out[0]
                            mean_sv=np.mean(sv, axis=1)
                            err_sv=(mean_sv-sv[:,0])
                            expl.list_sv_mean.append(mean_sv)
                            expl.list_sv_err.append(err_sv)

                    ground_sv=ground_model_imp.list_sv[IDX]
                    mean_ground_sv=ground_model_imp.list_sv_mean[IDX]
                    err_ground_sv=ground_model_imp.list_sv_err[IDX]
                    # err_ground_sv=(mean_ground_sv-ground_sv[:,0])
                    for k, mtr in metrics.items():
                        for k, expl in explainers_imp.items():
                            if expl.name != ground_model_imp.name:
                                if 'ShapleyRegression' in expl.name or 'R-MC' in expl.name:
                                    sv_comparison=expl.list_usv[IDX]
                                    mean_sv_comparison=expl.list_usv_mean[IDX]
                                    err_sv_comparison=expl.list_usv_err[IDX]

                                    if mtr.name == 'L2':
                                        res_mean=mtr.compute(mean_sv_comparison, mean_ground_sv)
                                        expl.list_ul2.append(res_mean)
                                        res_err=mtr.compute(err_sv_comparison, err_ground_sv)
                                        expl.list_ul2_err.append(res_err)
                                    if mtr.name == 'EuclideanDistance':
                                        res_distance=mtr.compute(sv_comparison, ground_sv)
                                        expl.list_udistance.append(res_distance)

                                sv_comparison=expl.list_sv[IDX]
                                mean_sv_comparison=expl.list_sv_mean[IDX]
                                err_sv_comparison=expl.list_sv_err[IDX]

                                if mtr.name == 'L2':
                                    res_mean=mtr.compute(mean_sv_comparison, mean_ground_sv)
                                    expl.list_l2.append(res_mean)
                                    res_err=mtr.compute(err_sv_comparison, err_ground_sv)
                                    expl.list_l2_err.append(res_err)
                                if mtr.name == 'EuclideanDistance':
                                    res_distance=mtr.compute(sv_comparison, ground_sv)
                                    expl.list_distance.append(res_distance)

                if self.ground_truth_imp!=None and self.ground_truth_ref!=None:
                    ground_sv_ref=ground_model_ref.list_sv[IDX]
                    ground_sv_imp=ground_model_imp.list_sv[IDX]
                    mtr=metrics['EuclideanDistance']
                    res=mtr.compute(ground_sv_ref, ground_sv_imp)
                    ground_model_ref.list_distance_versus.append(res)

            
            if verbose:
                print("-"*100)
            

    def print_results(self, dsfn):
        explainers_ref=self.explainers_reformulated_init[dsfn]
        explainers_imp=self.explainers_improved_init[dsfn]

        table = [['Method', 'Time', 'L2_M', 'L2_W', 'Euclidean']]
        for k, expl in explainers_ref.items():
            if 'ShapleyRegression' in expl.name:
                table.append(["R-UKS", np.mean(expl.list_time), np.mean(expl.list_ul2), 0, np.mean(expl.list_udistance)])
                table.append(["R-KS", np.mean(expl.list_time), np.mean(expl.list_l2), 0, np.mean(expl.list_distance)])
            else:
                if len(expl.list_l2)==0:
                    table.append([expl.name, np.mean(expl.list_time), 0, 0, 0])
                else:
                    table.append([expl.name, np.mean(expl.list_time), np.mean(expl.list_l2), 0, np.mean(expl.list_distance)])

        print(tabulate(table, headers='firstrow', headersglobalalign='center', tablefmt='fancy_grid', colalign=('center','center','global','global','global','global','global')))

        table = [['Method', 'Time', 'L2_M', 'L2_W', 'Euclidean']]
        for k, expl in explainers_imp.items():
            if 'ShapleyRegression' in expl.name:
                table.append(["I-UKS", np.mean(expl.list_time), np.mean(expl.list_ul2), np.mean(expl.list_ul2_err), np.mean(expl.list_udistance)])
                table.append(["I-KS", np.mean(expl.list_time),  np.mean(expl.list_l2), np.mean(expl.list_l2_err), np.mean(expl.list_distance)])
            elif 'R-MC' in expl.name:
                table.append(["I-MC", np.mean(expl.list_time), np.mean(expl.list_ul2), np.mean(expl.list_ul2_err), np.mean(expl.list_udistance)])
            else:
                if len(expl.list_l2)==0:
                    table.append([expl.name, np.mean(expl.list_time), 0, 0, 0])
                else:
                    table.append([expl.name, np.mean(expl.list_time), np.mean(expl.list_l2), np.mean(expl.list_l2_err), np.mean(expl.list_distance)])

        print(tabulate(table, headers='firstrow', headersglobalalign='center', tablefmt='fancy_grid', colalign=('center','center','global','global','global','global','global')))
