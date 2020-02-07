from utils import pre_processing, compute_sparse_correlation_matrix
import utils

import numpy as np
from scipy import io, sparse
from math import sqrt
from time import time



# Through all this code Rk_iu and Nk_iu are the same since implicit matrix is
#    made from the rating matrix without additional information (i.e. indexes of
#    non-zero elements are the same therefore neighbors too).


#################################################
# Naive way (iterate through each r_ui)
#################################################
def predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi):
  bui = mu + baseline_bu[u] + baseline_bi[0, i]
  buj = mu + baseline_bu[u] + baseline_bi[0, Sk_iu]
  return bui + 1 / S[i, Sk_iu].sum() * (S[i, Sk_iu].toarray().ravel() * (mat[u, Sk_iu].toarray().ravel() - buj)).sum()

def correlation_based_neighbourhood_model(mat, mat_file, l_reg2=100.0, k=250):
    # subsample the matrix to make computation faster
    mat = mat[0:mat.shape[0]//128, 0:mat.shape[1]//128]
    mat = mat[mat.getnnz(1)>0][:, mat.getnnz(0)>0]

    print(mat.shape)
    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    #baseline_bu, baseline_bi = baseline_estimator(mat)
    # We should call baseline_estimator but we can init at random for test
    baseline_bu, baseline_bi = np.random.rand(no_users, 1)  * 2 - 1, np.random.rand(1, no_movies) * 2 - 1    

    #bu_index, bi_index = pre_processing(mat, mat_file)

    mu = mat.data[:].mean()
    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)

    # Computation
    print("Computation...")
    n_iter = 200
    cx = mat.tocoo()
    for u,i,v in zip(cx.row, cx.col, cx.data):
        Sk_iu = np.flip(np.argsort(S[i,].toarray()))[:k].ravel()
        r_ui = predict_r_ui(mat, u, i, mu, S, Sk_iu, baseline_bu, baseline_bi)

#################################################


if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']
    correlation_based_neighbourhood_model(mat, mat_file)
