import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (interfere_with_anchors, get_measurements, 
                   quantise_measurements, do_MDS, ortho_procrustes, 
                   make_D_ensembles)

def y_hat_lam_forSR_LS(lam, A, D, b, f):
    return np.linalg.inv(A.T @ A + lam*D) @ (A.T @ b - lam*f)

def phi_lam_forSR_LS(y_hat, D, f):
    return y_hat.T @ D @ y_hat + 2 * f.T @ y_hat

def bisection_algo(D, A, b, f):
    evals, V  = np.linalg.eig(A.T @ A)
    evals = np.diag(evals**-0.5)
    ATA_half = V @ evals @ V.T
    
    eiglam, _  = np.linalg.eig(ATA_half @ D @ ATA_half)
    I_min = -1/np.max(eiglam)
    L = I_min.copy()
    R = np.random.uniform(1e-1, 1)
    
    n_iter = 200
    tol = 1e-13
    
    y_R = y_hat_lam_forSR_LS(R, A, D, b, f)
    f_R = phi_lam_forSR_LS(y_R, D, f)
    while(f_R > 0):
        R *=2
        y_R = y_hat_lam_forSR_LS(R, A, D, b, f)
        f_R = phi_lam_forSR_LS(y_R, D, f)
     
    for i in range(n_iter):        
        y_L = y_hat_lam_forSR_LS(L, A, D, b, f)
        f_L = phi_lam_forSR_LS(y_L, D, f)
        
        y_R = y_hat_lam_forSR_LS(R, A, D, b, f)
        f_R = phi_lam_forSR_LS(y_R, D, f)

        M = 0.5*(L+R)
        y_M = y_hat_lam_forSR_LS(M, A, D, b, f)
        f_M = phi_lam_forSR_LS(y_M, D, f)
        
        if (np.abs(f_M - 0) < tol):
            break
        
        if(np.sign(f_M) == np.sign(f_L)):
            L = M
        else:
            R = M
            
    return M

def SR_LS(anchor_positions, noisy_distances):
    X_anchors = np.hstack((np.real(anchor_positions), np.imag(anchor_positions))).T
    
    A = np.hstack((-2*X_anchors.T, np.ones([X_anchors.shape[1],1])))
    b = noisy_distances.reshape([-1,1]) - np.abs(anchor_positions)**2
    
    dim = X_anchors.shape[0]
    D = np.zeros([dim+1, dim+1])
    D[:dim, :dim] = np.eye(dim)
    
    f = np.zeros([dim+1, 1])
    f[-1] = -0.5
    
    lam = bisection_algo(D, A, b, f)
    y_hat = y_hat_lam_forSR_LS(lam, A, D, b, f)
    return y_hat[0] + 1j* y_hat[1]

###############################################################################
if __name__ == "__main__":
    np.random.seed(10)
    
    image_size = 64
    num_bits = 8
    trial = 0
    number_of_anchors = 15
    num_of_rows_in_A = 100
    
    x = np.random.normal(size=[1,image_size**2])
    anchors = np.random.normal(size=(number_of_anchors,image_size**2))
    opu_input = interfere_with_anchors(image_size, x, anchors)
    
    y, A = get_measurements(opu_input, num_of_rows_in_A)    
    y, y_quant = quantise_measurements(y, num_bits)
        
    all_D_oracle = make_D_ensembles(y, number_of_anchors)
    all_D_measured = make_D_ensembles(y_quant, number_of_anchors)
    
    rel_errors_single = np.zeros([num_of_rows_in_A, number_of_anchors-2+1])
    rel_errors_group = np.zeros([num_of_rows_in_A, number_of_anchors-2+1])

    for trial in tqdm(range(num_of_rows_in_A)):
        for i in range(2,number_of_anchors+1):
            ##### SINGLE POINT
            row_num = trial
            
            # true positions of everything after scaling
            b_true = (A @ opu_input.T).T
            scaling = np.linalg.norm(
                    b_true[0, row_num])**2 / all_D_oracle[row_num,0,1]
            b_anchors = (A @ anchors.T).T
            b_anchors /= scaling**0.5
            b_true /= scaling**0.5
            del b_true
            b_anchors = np.vstack((b_anchors, np.zeros(
                    [1, b_anchors.shape[1]]))) # add zero anchor
            
            x_hat_single = SR_LS(b_anchors[-(i+1):,row_num].reshape([-1,1]), 
                                           all_D_measured[row_num, 0, -(i+1):])
            
            x_true = (A @ x.T).T
            x_true /= scaling**0.5
            
            rel_errors_single[trial, i-2] = -20*np.log10(np.linalg.norm(
                    x_true[0,row_num] - x_hat_single) / np.linalg.norm(
                            x_true[0,row_num]))
            
            ##### MULTIPLE POINTS
            ind = np.arange(all_D_measured.shape[1] - (i+1),
                            all_D_measured.shape[1]).tolist()
            ind.append(0)
            ind.sort()
            D_quant = all_D_measured[:,:,ind][trial][ind,:]
            recovered_points = do_MDS(D_quant, i)
            fixed = np.hstack((x_true[0,row_num], b_anchors[-(i+1):,row_num]))
            recovered_points = ortho_procrustes(fixed, recovered_points)
            x_hat_group = recovered_points[0]
            rel_errors_group[trial, i-2] = -20*np.log10(np.linalg.norm(
                    x_true[0,row_num] - x_hat_group) / np.linalg.norm(
                            x_true[0,row_num]))
            
    anchors_axis = np.arange(2, number_of_anchors+1)
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5,4))
    plt.plot(anchors_axis, np.mean(rel_errors_single, axis=0), 
             label='Single point')
    plt.plot(anchors_axis, np.mean(rel_errors_group, axis=0), 
             label='Group')
    plt.ylabel('SNR (dB)')
    plt.xlabel('Number of anchors')
    plt.legend()
    plt.tight_layout()
