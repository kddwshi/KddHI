import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

from ..evaluation import Benchmark
from ..evaluation.utils import get_grand_null_u

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

class BarPlot():
    def __init__(self, benchmark, dsfn):
        self.benchmark = benchmark
        self.dsfn = dsfn

    def plot(self, index=None, type="reformulated"):
        width = 0.5
        if type == "reformulated":
            explainers=self.benchmark.explainers_reformulated_init[self.dsfn]
            colors_dict = {
                'R-EXT': 'tab:gray',
                'R-MC': 'tab:green',
                'R-UKS': 'tab:red',
                'R-KS': 'tab:orange',
            }
            shift_dict = {
                'R-EXT': -1*width/2.7,
                'R-UKS': -width/8,
                'R-KS': +width/8,
                'R-MC': +1*width/2.7,
            }
        elif type == "improved":
            explainers=self.benchmark.explainers_improved_init[self.dsfn]
            colors_dict = {
                'I-EXT': 'tab:gray',
                'I-MC': 'tab:green',
                'I-UKS': 'tab:red',
                'I-KS': 'tab:orange',
            }
            shift_dict = {
                'I-EXT': -1*width/2.7,
                'I-UKS': -width/8,
                'I-KS': +width/8,
                'I-MC': +1*width/2.7,
            }
        else:
            print('Invalid type')
            return
        
        self.dset = self.dsfn()
        X, Y, X_test, Y_test, feature_names, dataset=self.dset.get_data()
        if X_test is None:
            num_features = X.shape[1]
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
            # data=X_train
            ss = StandardScaler()
            ss.fit(X_train)
            X_train_ss = ss.transform(X_train)
            X_val_ss = ss.transform(X_val)
        else:
            num_features = X.shape[1]
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
            # data=X_train
            ss = StandardScaler()
            ss.fit(X_train)
            X_train_ss = ss.transform(X_train)
            X_val_ss = ss.transform(X_val)
            X_test_ss = ss.transform(X_test)

        # print(f'Feature names: {feature_names}')
        if index is not None and index >= len(X_train_ss):
            print('Sample index out of range')
            return

        plt.figure(figsize=(16, 10), dpi=300)
        

        if index == None:
            for k, expl in explainers.items():
                # print(f'Explainer: {k}')
                # print(np.mean(expl.list_sv_mean, axis=0))
                # plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv, axis=0), width=width/4, label=k, color=colors_dict[k])
                if type == "reformulated":
                    if 'ShapleyRegression' in k:
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-KS'], np.mean(expl.list_sv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_sv_err, axis=0), label="R-KS", color=colors_dict['R-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-UKS'], np.mean(expl.list_usv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_usv_err, axis=0), label='R-UKS', color=colors_dict['R-UKS'], capsize=5, linewidth=3)
                    else:
                        plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_sv_err, axis=0),label=k, color=colors_dict[k], capsize=5, linewidth=3)
                else:
                    if 'ShapleyRegression' in k:
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-KS'], np.mean(expl.list_sv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_sv_err, axis=0), label="I-KS", color=colors_dict['I-KS'],   capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-UKS'], np.mean(expl.list_usv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_usv_err, axis=0), label='I-UKS', color=colors_dict['I-UKS'], capsize=5, linewidth=3)
                    else:
                        if k=='R-MC':
                            plt.bar(np.arange(len(feature_names))+shift_dict['I-MC'], np.mean(expl.list_sv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_sv_err, axis=0), label=k, color=colors_dict['I-MC'], capsize=5, linewidth=3)
                        else:
                            plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv_mean, axis=0), width=width/4, yerr=np.mean(expl.list_sv_err, axis=0), label=k, color=colors_dict[k], capsize=5, linewidth=3)
                
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'figure/barplot_{dataset}.pdf', dpi=300)
            plt.show()
        else: 
            sample = X_train_ss[index]
            sample= np.array([sample]) # add dimension to sample
            
            label = Y_train[index]
            IDX = index

            x_t = torch.tensor(sample, dtype=torch.float32, device='cpu')
            # print(x_t, label, self.benchmark.num_features[self.dsfn])
            grand_u, null_u, diff_u = get_grand_null_u(x_t, self.benchmark.num_features[self.dsfn], label, self.benchmark.surrogate_VV[self.dsfn])

            # print(sample, label)
            # print(diff_u)
            for k, expl in explainers.items():
                out=expl.compute(IDX, sample, label, self.benchmark.kernelshap_iters, diff_u)
                # print(k, out)
                mean_sv = np.mean(out[0], axis=1)
                err_sv = (mean_sv - out[0][:,0])
                if type == "reformulated":
                    if 'ShapleyRegression' in k:
                        mean_sv2 = np.mean(out[1], axis=1)
                        err_sv2 = (mean_sv2 - out[1][:,0])
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-UKS'], mean_sv, width=width/4, yerr=err_sv, label="R-UKS", color=colors_dict['R-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-KS'], mean_sv2, width=width/4, yerr=err_sv2, label='R-KS', color=colors_dict['R-UKS'], capsize=5, linewidth=3)
                    else:
                        plt.bar(np.arange(len(feature_names))+shift_dict[k], mean_sv, width=width/4, yerr=err_sv, label=k, color=colors_dict[k], capsize=5, linewidth=3)
                else:
                    if 'ShapleyRegression' in k:
                        mean_sv2 = np.mean(out[1], axis=1)
                        err_sv2 = (mean_sv2 - out[1][:,0])
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-KS'], mean_sv, width=width/4, yerr=err_sv, label="I-KS", color=colors_dict['I-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-UKS'], mean_sv2, width=width/4, yerr=err_sv2, label='I-UKS', color=colors_dict['I-UKS'], capsize=5, linewidth=3)
                    else:
                        if k=='R-MC':
                            plt.bar(np.arange(len(feature_names))+shift_dict['I-MC'], mean_sv, width=width/4, yerr=err_sv, label=k, color=colors_dict['I-MC'], capsize=5, linewidth=3)
                        else:
                            plt.bar(np.arange(len(feature_names))+shift_dict[k], mean_sv, width=width/4, yerr=err_sv, label=k, color=colors_dict[k], capsize=5, linewidth=3)
            

            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'figure/barplot_{dataset}.pdf', dpi=300)
            plt.show()


class CoVPlot():
    def __init__(self, benchmark, dsfn):
        self.benchmark = benchmark
        self.dsfn = dsfn

    def plot(self, index=None, type="reformulated"):
        width = 0.5
        if type == "reformulated":
            explainers=self.benchmark.explainers_reformulated_init[self.dsfn]
            colors_dict = {
                'R-EXT': 'tab:gray',
                'R-MC': 'tab:green',
                'R-UKS': 'tab:red',
                'R-KS': 'tab:orange',
            }
            shift_dict = {
                'R-EXT': -1*width/2.7,
                'R-UKS': -width/8,
                'R-KS': +width/8,
                'R-MC': +1*width/2.7,
            }
        elif type == "improved":
            explainers=self.benchmark.explainers_improved_init[self.dsfn]
            colors_dict = {
                'I-EXT': 'tab:gray',
                'I-MC': 'tab:green',
                'I-UKS': 'tab:red',
                'I-KS': 'tab:orange',
            }
            shift_dict = {
                'I-EXT': -1*width/2.7,
                'I-UKS': -width/8,
                'I-KS': +width/8,
                'I-MC': +1*width/2.7,
            }
        else:
            print('Invalid type')
            return
        
        self.dset = self.dsfn()
        X, Y, X_test, Y_test, feature_names, dataset=self.dset.get_data()
        if X_test is None:
            num_features = X.shape[1]
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
            # data=X_train
            ss = StandardScaler()
            ss.fit(X_train)
            X_train_ss = ss.transform(X_train)
            X_val_ss = ss.transform(X_val)
        else:
            num_features = X.shape[1]
            X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=0)
            # data=X_train
            ss = StandardScaler()
            ss.fit(X_train)
            X_train_ss = ss.transform(X_train)
            X_val_ss = ss.transform(X_val)
            X_test_ss = ss.transform(X_test)

        # print(f'Feature names: {feature_names}')
        if index is not None and index >= len(X_train_ss):
            print('Sample index out of range')
            return

        plt.figure(figsize=(16, 10), dpi=300)
        

        if index == None:
            for k, expl in explainers.items():
                # print(f'Explainer: {k}')
                # print(np.mean(expl.list_sv_mean, axis=0))
                # plt.bar(np.arange(len(feature_names))+shift_dict[k], np.mean(expl.list_sv, axis=0), width=width/4, label=k, color=colors_dict[k])
                if type == "reformulated":
                    if 'ShapleyRegression' in k:
                        var=np.mean(expl.list_sv_err, axis=0)/np.abs(np.mean(expl.list_sv_mean, axis=0))
                        var1=np.mean(expl.list_usv_err, axis=0)/np.abs(np.mean(expl.list_usv_mean, axis=0))
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-KS'], var, width=width/4, label="R-KS", color=colors_dict['R-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-UKS'], var1, width=width/4, label='R-UKS', color=colors_dict['R-UKS'], capsize=5, linewidth=3)
                    else:
                        var=np.mean(expl.list_sv_err, axis=0)/np.abs(np.mean(expl.list_sv_mean, axis=0))
                        plt.bar(np.arange(len(feature_names))+shift_dict[k], var, width=width/4, label=k, color=colors_dict[k], capsize=5, linewidth=3)
                else:
                    if 'ShapleyRegression' in k:
                        var=np.mean(expl.list_sv_err, axis=0)/np.abs(np.mean(expl.list_sv_mean, axis=0))
                        var1=np.mean(expl.list_usv_err, axis=0)/np.abs(np.mean(expl.list_usv_mean, axis=0))
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-KS'], var, width=width/4, label="I-KS", color=colors_dict['I-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-UKS'], var1, width=width/4, label='I-UKS', color=colors_dict['I-UKS'], capsize=5, linewidth=3)
                    else:
                        if k=='R-MC':
                            var=np.mean(expl.list_sv_err, axis=0)/np.abs(np.mean(expl.list_sv_mean, axis=0))
                            plt.bar(np.arange(len(feature_names))+shift_dict['I-MC'], var, width=width/4, label=k, color=colors_dict['I-MC'], capsize=5, linewidth=3)
                        else:
                            var=np.mean(expl.list_sv_err, axis=0)/np.abs(np.mean(expl.list_sv_mean, axis=0))
                            plt.bar(np.arange(len(feature_names))+shift_dict[k], var, width=width/4, label=k, color=colors_dict[k], capsize=5, linewidth=3)
            
            plt.axhline(y = 1, color = 'blue', linestyle = '--')
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'figure/cov_{dataset}.pdf', dpi=300)
            plt.show()
        else: 
            sample = X_train_ss[index]
            sample= np.array([sample]) # add dimension to sample
            
            label = Y_train[index]
            IDX = index

            x_t = torch.tensor(sample, dtype=torch.float32, device='cpu')
            grand_u, null_u, diff_u = get_grand_null_u(x_t, self.benchmark.num_features[self.dsfn], label, self.benchmark.surrogate_VV[self.dsfn])

            # print(sample, label)
            # print(diff_u)
            for k, expl in explainers.items():
                out=expl.compute(IDX, sample, label, self.benchmark.kernelshap_iters, diff_u)
                # print(k, out)
                mean_sv = np.mean(out[0], axis=1)
                err_sv = (mean_sv - out[0][:,0])
                var=err_sv/np.abs(mean_sv)
                if type == "reformulated":
                    if 'ShapleyRegression' in k:
                        mean_sv2 = np.mean(out[1], axis=1)
                        err_sv2 = (mean_sv2 - out[1][:,0])
                        var2=err_sv2/np.abs(mean_sv2)

                        plt.bar(np.arange(len(feature_names))+shift_dict['R-UKS'], var, width=width/4, label="R-UKS", color=colors_dict['R-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['R-KS'], var2, width=width/4, label='R-KS', color=colors_dict['R-UKS'], capsize=5, linewidth=3)
                    else:
                        plt.bar(np.arange(len(feature_names))+shift_dict[k], var, width=width/4, label=k, color=colors_dict[k], capsize=5, linewidth=3)
                else:
                    if 'ShapleyRegression' in k:
                        mean_sv2 = np.mean(out[1], axis=1)
                        err_sv2 = (mean_sv2 - out[1][:,0])
                        var2=err_sv2/np.abs(mean_sv2)
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-KS'], var, width=width/4, label="I-KS", color=colors_dict['I-KS'], capsize=5, linewidth=3)
                        plt.bar(np.arange(len(feature_names))+shift_dict['I-UKS'], var2, width=width/4, label='I-UKS', color=colors_dict['I-UKS'], capsize=5, linewidth=3)
                    else:
                        if k=='R-MC':
                            plt.bar(np.arange(len(feature_names))+shift_dict['I-MC'], var, width=width/4, label=k, color=colors_dict['I-MC'], capsize=5, linewidth=3)
                        else:
                            plt.bar(np.arange(len(feature_names))+shift_dict[k], var, width=width/4, label=k, color=colors_dict[k], capsize=5, linewidth=3)
            
            plt.axhline(y = 1, color = 'blue', linestyle = '--')
            plt.legend(fontsize=20)
            plt.tick_params(labelsize=18)
            plt.ylabel('SHAP Values', fontsize=18)
            # plt.title('BarPlot', fontsize=18)
            plt.xticks(np.arange(len(feature_names)), feature_names, rotation=35, rotation_mode='anchor', ha='right')

            plt.tight_layout()
            # save figure
            plt.savefig(f'figure/cov_{dataset}_{index}.pdf', dpi=300)
            plt.show()


class TimeFeaturePlot():
    def __init__(self, benchmark):
        self.benchmark = benchmark

    def plot(self):
        dict={}
        dict_expl={}
        for dset_fn in self.benchmark.dataset:
            dset=dset_fn()
            X, Y, X_test, Y_test, feature_names, dataset=dset.get_data()
            explainers_ref=self.benchmark.explainers_reformulated_init[dset_fn]
            res={}
            for k, expl in explainers_ref.items():
                if expl.name == "R-MC":
                    res["MC"] = np.mean(expl.list_time)
                    if expl.name not in dict_expl:
                        dict_expl["MC"] = []
                else:
                    res[expl.name] = np.mean(expl.list_time)
                    if expl.name not in dict_expl:
                        dict_expl[expl.name] = []

            explainers_imp=self.benchmark.explainers_improved_init[dset_fn]
            for k, expl in explainers_imp.items():
                if expl.name != "R-MC":
                    res[expl.name] = np.mean(expl.list_time)
                    if expl.name not in dict_expl:
                        dict_expl[expl.name] = []
            
            num_features=len(feature_names)
            dict[num_features] = res
        
        dict = {k: v for k, v in sorted(dict.items(), key=lambda item: item[0])}
        xticks = [k for k,v in dict.items()]
        for k,v in dict.items():
            for expl, time in v.items():
                dict_expl[expl].append(time)
        
        plt.figure(figsize=(16, 10), dpi=300)
        color_list=["olive", "green", "orange", "brown", "red", "purple", "blue", "cyan"]
        i=0
        for expl, times in dict_expl.items():
            # print(np.arange(1,len(times)), times)
            plt.plot(np.arange(1,len(times)+1), times, label=expl, color=color_list[i], linewidth=2.5)
            i+=1

        plt.xticks(np.arange(1,len(times)+1), xticks)

        # set y-axis to log scale
        plt.yscale('log')
        # Annotations
        plt.legend(fontsize=16, loc='upper left')
        plt.tick_params(labelsize=16)
        plt.xlabel('Number of Featuers', fontsize=18)
        plt.ylabel('Time (s)', fontsize=18)
        # plt.title('TimeFeature', fontsize=18)
        plt.tight_layout()
        # save the plot
        plt.savefig(f'figure/timefeature.pdf', dpi=300)
        plt.show()

    