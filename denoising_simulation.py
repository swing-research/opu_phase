import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_D(max_anchors):
    n = 100 # signal length
    k = 1    # number of signals to recover
    m = max_anchors    # number of reference signals
    
    a = np.random.randn(n, 1) + 1j*np.random.randn(n, 1)
    x = np.random.randn(n, k) # signal of interest
    w = np.random.randn(n, m) # anchors
    
    D = np.zeros((m + 2, m + 2)) # empty EDM
    
    # fill the EDM with all sorts of distances
    for i in range(m):
        # compute distances between the refs and the target signal
        D[0, i + 1] = np.abs(np.dot(a.T, x[:, 0] - w[:, i]))**2
        D[i + 1, 0] = D[0, i + 1]
    
        # add distances to the origin (point m+1) to avoid translation ambiguity
        D[m + 1, i + 1] = np.abs(np.dot(a.T, w[:, i]))**2
        D[i + 1, m + 1] = D[m + 1, i + 1]
    
        # compute all inter-ref distances
        for j in range(m):
            D[i + 1, j + 1] = np.abs(np.dot(a.T, w[:, i] - w[:, j]))**2
    
    # add distance between the target and the origin 
    D[0, m + 1] = np.abs(np.dot(a.T, x[:, 0]))**2
    D[m + 1, 0] = D[0, m + 1]
    
    return D

def do_MDS(D, number_of_anchors):
    m = number_of_anchors
    J = np.eye(m + 2) - 1. / (m + 2) * np.ones((m + 2, m + 2))
    G = -1/2 * np.dot(J, D).dot(J)
    U, s, VT = np.linalg.svd(G)
    Z_est_R2 = np.dot(np.diag(np.sqrt(s[:2])), VT[:2, :])
    Z_est_cpx = Z_est_R2[0, :] + 1j*Z_est_R2[1, :]
    
    # translate the origin back at (0, 0)
    Z_est_cpx -= Z_est_cpx[m + 1]
    
    return Z_est_cpx
    
def make_LXX(X):
    e = np.ones([X.shape[1],1])
    G = X.T @ X
    diag_vec = np.diag(G).reshape([-1,1])
    L =  diag_vec @ e.T + e @ diag_vec.T - 2*G
    return L
    
def gradient_descent_X(D, X_0, W):
    lr = 0.005
    if (np.isnan(lr)):
        lr = 0
    if (np.isinf(lr)):
        lr = 0
    n_iter = 30
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

def check_denoising():
    trials = 100
    max_anchors = 15
    all_D_oracles = np.zeros([trials, max_anchors + 2, max_anchors + 2])
    
    x_snrs = np.zeros([trials, 3, max_anchors+1-2])
    
    for i in range(trials):
        D_oracle = get_D(max_anchors)
        all_D_oracles[i] = D_oracle

    n_bits = 8
    floor = 0
    all_D_oracles /= (1.0*np.max(all_D_oracles))
    all_D_oracles *= (2**n_bits - 1)
    quant_D_oracles = np.round(all_D_oracles)
    
    quant_D_oracles[quant_D_oracles<floor] = 0
    
    for trial in tqdm(range(trials)):
        for i in range(2,max_anchors+1):
            ind = np.arange(i+1).tolist()
            ind.append(max_anchors+1)
            D_quant = quant_D_oracles[:,:,ind][trial][ind,:]
            recovered_points = do_MDS(D_quant, i)
            D_recon = make_LXX(np.vstack((np.real(recovered_points), np.imag(recovered_points))))
            
            X_0 = np.vstack((np.real(recovered_points), np.imag(recovered_points)))
            W = (D_quant>0).astype('float') + np.eye(D_quant.shape[0])
            X, _ = gradient_descent_X(D_quant, X_0, W)
            recovered_points = X[0] + 1j*X[1]
            D_recon_2 = make_LXX(X)
            
            D_original = all_D_oracles[:,:,ind][trial][ind,:]
            
            x_snrs[trial, 0,i-2] = -20*np.log10(np.abs(D_original[0,-1]-D_quant[0,-1])/(D_original[0,-1]))
            x_snrs[trial, 1,i-2] = -20*np.log10(np.abs(D_original[0,-1]-D_recon[0,-1])/(D_original[0,-1]))
            x_snrs[trial, 2,i-2] = -20*np.log10(np.abs(D_original[0,-1]-D_recon_2[0,-1])/(D_original[0,-1]))
    
    x_bits = np.mean(x_snrs, axis=0) / 6.02
    
    anchors = np.arange(2, max_anchors+1)
    
    plt.rcParams.update({'font.size': 14})
    plt.figure(figsize=(5,4))
    plt.plot(anchors, x_bits[0], label='Measured')
    plt.plot(anchors, x_bits[1], label='MDS')
    plt.plot(anchors, x_bits[2], label='MDS-GD')
    plt.ylabel('Average good bits')
    plt.xlabel('Number of anchors')
    plt.legend()
    plt.tight_layout()

###############################################################################
if __name__ == "__main__":
    np.random.seed(3)
    check_denoising()