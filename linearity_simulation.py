import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (interfere_with_anchors, get_measurements, 
                   quantise_measurements, do_MDS, ortho_procrustes, make_LXX)

def gradient_descent_X(D, X_0, W):
    lr = 0.007 
    if (np.isnan(lr)):
        lr = 0
    if (np.isinf(lr)):
        lr = 0
    n_iter = 20
    lam = 0
    
    N = X_0.shape[1]
    e = np.ones([N,1])
    
    X = X_0.copy()
    
    for i in range(n_iter):
        L = make_LXX(X)
        P = D - L
        P = W*P
        grad = (1/N**2) * (8 * X @ (P - np.diag(np.diag(P @ e)) ) + lam*2*X)
        X -= lr*grad

    return X, L

def make_D_ensembles(y, number_of_anchors):
    num_elements = int((number_of_anchors+2)* (number_of_anchors+1) * 0.5)
    
    trials = y.shape[1]
    dim = number_of_anchors+2
    all_D_oracles_x1 = np.zeros([trials, dim, dim])
    all_D_oracles_x2 = np.zeros([trials, dim, dim])
    all_D_oracles_x1_plus_x2 = np.zeros([trials, dim, dim])
    
    ind = np.triu_indices(all_D_oracles_x1[0].shape[0], k=1)
    for i in range(trials):
        data = y[0:num_elements,i]
        all_D_oracles_x1[i][ind] = data
        all_D_oracles_x1[i] += all_D_oracles_x1[i].T
        
        data = y[num_elements: 2*num_elements,i]
        all_D_oracles_x2[i][ind] = data
        all_D_oracles_x2[i] += all_D_oracles_x2[i].T
        
        data = y[2*num_elements: 3*num_elements,i]
        all_D_oracles_x1_plus_x2[i][ind] = data
        all_D_oracles_x1_plus_x2[i] += all_D_oracles_x1_plus_x2[i].T
        
    return all_D_oracles_x1, all_D_oracles_x2, all_D_oracles_x1_plus_x2
    

###############################################################################
if __name__ == "__main__":
    np.random.seed(5) # 4 is nice too
    
    image_size = 64
    number_of_anchors = 15
    num_of_rows_in_A = 100
    num_bits = 8
    
    xs = np.random.normal(size=(3,image_size**2))
    xs[2] = xs[0]+xs[1]
    anchors = np.random.normal(size=(number_of_anchors,image_size**2))

    x1_inter = interfere_with_anchors(image_size, xs[0], anchors)
    x2_inter = interfere_with_anchors(image_size, xs[1], anchors)
    x1_plus_x2_inter = interfere_with_anchors(image_size, xs[2], anchors)
    
    opu_input = np.vstack((x1_inter, x2_inter, x1_plus_x2_inter))
    
    y, A = get_measurements(opu_input, num_of_rows_in_A)
    y, y_quant = quantise_measurements(y, num_bits)
    floor = 0
    y_quant[y_quant<floor] = 0
    
    all_D_quant_x1, all_D_quant_x2, all_D_quant_x1_plus_x2 = make_D_ensembles(
            y_quant, number_of_anchors)
    
    manual = np.zeros([num_of_rows_in_A, number_of_anchors+1-2]).astype(
            'complex128')
    direct = np.zeros([num_of_rows_in_A, number_of_anchors+1-2]).astype(
            'complex128')
    
    manual_gd = np.zeros([num_of_rows_in_A, number_of_anchors+1-2]).astype(
            'complex128')
    direct_gd = np.zeros([num_of_rows_in_A, number_of_anchors+1-2]).astype(
            'complex128')
    
    for trial in tqdm(range(num_of_rows_in_A)):
        for i in range(2,number_of_anchors+1):
            
            ind = np.random.choice(np.arange(1, number_of_anchors+1), i, 
                                   replace=False)
            ind = np.hstack((ind,0,number_of_anchors+1))
            ind.sort()
            
            D_quant_x1 = all_D_quant_x1[:,:,ind][trial][ind,:]
            D_quant_x2 = all_D_quant_x2[:,:,ind][trial][ind,:]
            D_quant_x1_plus_x2 = all_D_quant_x1_plus_x2[:,:,ind][trial][ind,:]
            
            # normal MDS
            recovered_points_x1 = do_MDS(D_quant_x1, i)
            recovered_points_x2 = do_MDS(D_quant_x2, i)
            recovered_points_x1_plus_x2 = do_MDS(D_quant_x1_plus_x2, i)
            recovered_points_x2 = ortho_procrustes(
                    recovered_points_x1, recovered_points_x2)
            recovered_points_x1_plus_x2 = ortho_procrustes(
                    recovered_points_x1, recovered_points_x1_plus_x2)
            
            manual_sum = recovered_points_x1[0] + recovered_points_x2[0]
            opu_sum = recovered_points_x1_plus_x2[0]
            manual[trial, i-2] = manual_sum
            direct[trial, i-2] = opu_sum
            
            # grad descent
            X_0 = np.vstack((np.real(recovered_points_x1), np.imag(
                    recovered_points_x1)))
            W = (D_quant_x1>0).astype('float') + np.eye(D_quant_x1.shape[0])
            X, L = gradient_descent_X(D_quant_x1, X_0, W)
            recovered_points_x1 = X[0] + 1j*X[1]
            recovered_points_x1 -= recovered_points_x1[-1]
            X_0 = np.vstack((np.real(recovered_points_x2), 
                             np.imag(recovered_points_x2)))
            W = (D_quant_x2>0).astype('float') + np.eye(D_quant_x2.shape[0])
            X, L = gradient_descent_X(D_quant_x2, X_0, W)
            recovered_points_x2 = X[0] + 1j*X[1]
            recovered_points_x2 -= recovered_points_x2[-1]
            X_0 = np.vstack((np.real(recovered_points_x1_plus_x2), 
                             np.imag(recovered_points_x1_plus_x2)))
            W = (D_quant_x1_plus_x2>0).astype('float') + np.eye(
                    D_quant_x1_plus_x2.shape[0])
            X, L = gradient_descent_X(D_quant_x1_plus_x2, X_0, W)
            recovered_points_x1_plus_x2 = X[0] + 1j*X[1]
            recovered_points_x1_plus_x2 -= recovered_points_x1_plus_x2[-1]
            
            recovered_points_x2 = ortho_procrustes(
                    recovered_points_x1, recovered_points_x2)
            recovered_points_x1_plus_x2 = ortho_procrustes(
                    recovered_points_x1, recovered_points_x1_plus_x2)
 
            manual_sum = recovered_points_x1[0] + recovered_points_x2[0]
            opu_sum = recovered_points_x1_plus_x2[0]
            manual_gd[trial, i-2] = manual_sum
            direct_gd[trial, i-2] = opu_sum
    
    anchors_axis = np.arange(2, number_of_anchors+1)
    rel_errors = 100*(np.abs(manual - direct) / np.abs(direct))
    rel_errors_sdr = 100*(np.abs(manual_gd - direct_gd) / np.abs(direct_gd))
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5,4))
    plt.plot(anchors_axis, np.mean(rel_errors, axis=0), label='MDS')
    plt.plot(anchors_axis, np.mean(rel_errors_sdr, axis=0), label='MDS-GD')
    plt.ylabel('Average relative error (%)')
    plt.xlabel('Number of anchors')
    plt.legend()
    plt.tight_layout()
    