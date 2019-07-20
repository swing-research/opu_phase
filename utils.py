import numpy as np

def complex_gaussian(shape):
    A = np.random.normal(size=(shape[0],shape[1],2)).astype('complex128')
    A[:,:,1] *= 1j
    A = np.sum(A, axis=2)
    return A

def interfere_with_anchors(n, x, anchors):
    interfered = anchors - x
    interfered = np.vstack((interfered, x)) # x with zero
    
    anchors = np.vstack((anchors, np.zeros(n**2))) # zero is an anchor too
    
    for i in range(anchors.shape[0]-1):
        diffs = anchors[i] - anchors[1+i:]
        interfered = np.vstack((interfered, diffs))
    
    return interfered

def get_measurements(opu_input, num_rand_proj):
    A = complex_gaussian([num_rand_proj, opu_input.shape[1]])
    measurement = (A @ opu_input.T).T 
    measurement = np.abs(measurement)**2
    
    return measurement, A

def quantise_measurements(y, n_bits):
    levels = (2**n_bits - 1)
    y = y / (1.0*np.max(y))
    y = y*levels
    y_quant = np.round(y)
    
    return y, y_quant

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

def ortho_procrustes(fixed, modify):
    fixed = np.vstack ((np.real(fixed[1:]), np.imag(fixed[1:])))
    modify = np.vstack ((np.real(modify), np.imag(modify)))
    original = modify.copy()
    modify = modify[:,1:]
    fixed_mean = (np.mean(fixed, axis=1)).reshape([-1,1])
    fixed -= fixed_mean
    modify_mean = (np.mean(modify, axis=1)).reshape([-1,1])
    modify -= modify_mean
    M = fixed @ modify.T
    u, s, vh = np.linalg.svd(M)
    R = u @ vh
    original = R @ (original - modify_mean @ np.ones([1, original.shape[1]])) + fixed_mean@np.ones([1, original.shape[1]])
    return original[0] + 1j*original[1]

def make_LXX(X):
    e = np.ones([X.shape[1],1])
    G = X.T @ X
    diag_vec = np.diag(G).reshape([-1,1])
    L =  diag_vec @ e.T + e @ diag_vec.T - 2*G

    return L

def make_D_ensembles(y, number_of_anchors):
    num_elements = int((number_of_anchors+2)* (number_of_anchors+1) * 0.5)
    
    trials = y.shape[1]
    dim = number_of_anchors+2
    all_D_oracles_x = np.zeros([trials, dim, dim])
    
    ind = np.triu_indices(all_D_oracles_x[0].shape[0], k=1)
    for i in range(trials):
        data = y[0:num_elements,i]
        all_D_oracles_x[i][ind] = data
        all_D_oracles_x[i] += all_D_oracles_x[i].T
        
    return all_D_oracles_x