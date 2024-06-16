import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn

def MonteCarlo_UNI(IND, DATA, SURROGATE, M):

    sv_m_mc=[]
    sv_u_mc=[]

    x=DATA.iloc[IND]
    ones = torch.ones(1, x.shape[0], dtype=torch.float32)

    # print("MC DATA:",x.values)

    link = nn.Softmax(dim=-1)

    for j in range(x.shape[0]):
        #M = 750 #####################################################################################################################################################################
        n_features = len(x)
        marginal_contributions = []
        marginal_contributions_U =[]

        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)
        for itr in range(M):
            z = DATA.sample(1).values[0]
            x_idx = random.sample(feature_idxs, min(max(int(0.2*n_features), random.choice(feature_idxs)), int(0.8*n_features))) #estraggo 0.8*feature_idx
            z_idx = [idx for idx in feature_idxs if idx not in x_idx] # features non estratte. Ricorda che una feauture, quella su cui si calcola lo SV, è sempre esclusa

            # construct two new instances
            x_plus_j = np.array([x[i] if i in x_idx + [j] else z[i] for i in range(n_features)])
            x_minus_j = np.array([z[i] if i in z_idx + [j] else x[i] for i in range(n_features)])

            ##############################################################################à
            # calculate marginal contribution
            x_plus_j=x_plus_j.reshape(1, -1)#np.expand_dims(x_plus_j, axis=0)
            x_minus_j=x_minus_j.reshape(1, -1)#np.expand_dims(x_minus_j, axis=0)
            # print(x_plus_j)
            # print(x_minus_j)


            # TODO
            # MARGINAL CONTRIBUTION FROM THE SURROGATE
            v1, v2 = SURROGATE(torch.tensor(x_plus_j, dtype=torch.float32), ones)
            v1 = link(v1[0])
            v2 = link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            # v1, v2 = OM.get_pred_VV(x_plus_j)
            # v1 = v1[0]
            # v2 = v2[0]
            # print(v1, v2)
            v_m_plus = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
            v_m_plus_U = np.array([np.abs(v1[0]-v2[1])/2, np.abs(v2[0]-v1[1])/2])

            v1, v2 = SURROGATE(torch.tensor(x_minus_j, dtype=torch.float32), ones)
            v1 = link(v1[0])
            v2 = link(v2[0])
            v1 = v1.data.numpy()
            v2 = v2.data.numpy()
            # v1, v2 = OM.get_pred_VV(x_minus_j)
            # v1 = v1[0]
            # v2 = v2[0]
            # print(v1, v2)
            v_m_minus = np.array([(v1[0]+v2[1])/2, (v2[0]+v1[1])/2])
            v_m_minus_U = np.array([np.abs(v1[0]-v2[1])/2, np.abs(v2[0]-v1[1])/2])
            
            # print(v_m_plus)
            # print(v_m_minus)

            marginal_contribution = v_m_plus - v_m_minus 
            marginal_contributions.append(marginal_contribution)
            
            marginal_contribution_U = np.abs(v_m_plus_U - v_m_minus_U)   # NO MARGINAL CONTRIBUTUION NEGATIVE?
            marginal_contributions_U.append(marginal_contribution_U)
            # break

        marginal_contributions=np.array(marginal_contributions)
        marginal_contributions_U=np.array(marginal_contributions_U)

        phi_j_x = np.sum(marginal_contributions, axis=0) / len(marginal_contributions)  # our shaply value
        phi_j_x_U = np.sum(marginal_contributions_U, axis=0) / len(marginal_contributions_U)  # our shaply value
        # break

        sv_m_mc.append(phi_j_x)
        sv_u_mc.append(phi_j_x_U)
        
    return np.array(sv_m_mc), np.array(sv_u_mc)