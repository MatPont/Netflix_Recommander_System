from utils import pre_processing, compute_sparse_correlation_matrix
import utils

import numpy as np
from scipy import io, sparse
from math import sqrt, isnan



#################################################
# Non-vectorized way
#################################################
def predict_r_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj):
    N_u_sum = yj[N_u].sum(0)
    return mu + bu[u] + bi[0, i] + np.dot(qi[i], (pu[u] + N_u_sum / sqrt(len(N_u))))

def compute_e_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj):
    return mat[u, i] - predict_r_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj)

def compute_loss(mat, mu, bu, bi, qi, pu, N_u, yj, l_reg6=0.005, l_reg7=0.015):
    loss = 0
    cx = mat.tocoo()
    for u,i,v in zip(cx.row, cx.col, cx.data):
        r_ui_pred = predict_r_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj)
        loss += (mat[u, i] - r_ui_pred) ** 2 + l_reg6 * ((bu ** 2).sum() + (bi ** 2).sum())
        loss += l_reg7 * ((qi[i]**2).sum() + (pu[u]**2).sum() + (yj[N_u]**2).sum())

    return loss

def svd_more_more(mat, mat_file, gamma1=0.007, gamma2=0.007, gamma3=0.001, l_reg2=100, l_reg6=0.005, l_reg7=0.015, f=50):
    # subsample the matrix to make computation faster
    """mat = mat[0:mat.shape[0]//128, 0:mat.shape[1]//128]
    mat = mat[mat.getnnz(1)>0][:, mat.getnnz(0)>0]"""

    print(mat.shape)
    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    bu_index, bi_index = pre_processing(mat, mat_file)
    
    # Init parameters
    bu = np.random.rand(no_users, 1)  * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    qi = np.random.rand(no_movies, f) * 2 - 1
    pu = np.random.rand(no_users, f) * 2 - 1
    yj = np.random.rand(no_movies, f) * 2 - 1

    mu = mat.data[:].mean()

    # Train
    print("Train...")
    n_iter = 200
    cx = mat.tocoo()
    for it in range(n_iter):
        for u,i,v in zip(cx.row, cx.col, cx.data):
            N_u = bi_index[u]
            e_ui = compute_e_ui(mat, u, i, mu, bu, bi, qi, pu, N_u, yj)

            bu[u] += gamma1 * (e_ui - l_reg6 * bu[u])
            bi[0, i] += gamma1 * (e_ui - l_reg6 * bi[0, i])
            qi[i] += gamma2 * (e_ui * (pu[u] + 1 / sqrt(len(N_u)) * yj[N_u].sum(0)) - l_reg7 * qi[i])
            pu[u] += gamma2 * (e_ui * qi[i] - l_reg7 * pu[u])
            yj[N_u] += gamma2 * (e_ui * 1/ sqrt(len(N_u)) * qi[i] - l_reg7 * yj[N_u])
        gamma1 *= 0.9
        gamma2 *= 0.9

        if it % 10 == 0:
          print(it, "\ ", n_iter)         
          print("compute loss...")
          print(compute_loss(mat, mu, bu, bi, qi, pu, N_u, yj, l_reg6=l_reg6, l_reg7=l_reg7))
    
    return bu, bi, qi, pu, yj
#################################################


if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']
    svd_more_more(mat, mat_file)
