from utils import pre_processing, compute_sparse_correlation_matrix
import numpy as np
from scipy import io, sparse
from math import sqrt, isnan



# Through all this code Rk_iu and Nk_iu are the same since implicit matrix is
#    made from the rating matrix without additional information.


#################################################
# Naive way
#################################################
def predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj):
    Rk_iu_sum = Nk_iu_sum = N_u_sum = 0
    for j in Rk_iu:
        buj = mu + baseline_bu[u] + baseline_bi[0, j]
        Rk_iu_sum += (mat[u, j] - buj) * wij[i][j]
        Nk_iu_sum += cij[i][j]
    N_u_sum = yj[N_u].sum(0)
    return mu + bu[u] + bi[0, i] + np.dot(qi[i], (pu[u] + N_u_sum / sqrt(len(N_u)))) + Rk_iu_sum / sqrt(len(Rk_iu)) + Nk_iu_sum / sqrt(len(Nk_iu))

def compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj):
    return mat[u, i] - predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)

def compute_loss(mat, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj, l_reg6=0.005, l_reg7=0.015, l_reg8=0.015):
    loss = 0
    cx = mat.tocoo()
    for u,i,v in zip(cx.row, cx.col, cx.data):
        r_ui_pred = predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)
        Rk_iu_sum = Nk_iu_sum = 0
        for j in Rk_iu:
            Rk_iu_sum += wij[i][j] ** 2
            Nk_iu_sum += cij[i][j] ** 2
        loss += (mat[u, i] - r_ui_pred) ** 2 + l_reg6 * ((bu ** 2).sum() + (bi ** 2).sum()) + l_reg8 * (Rk_iu_sum + Nk_iu_sum)
        loss += l_reg7 * ((qi[i]**2).sum() + (pu[u]**2).sum() + (yj[N_u]**2).sum())

    return loss

def integrated_model(mat, mat_file, gamma1=0.007, gamma2=0.007, gamma3=0.001, l_reg2=100, l_reg6=0.005, l_reg7=0.015, l_reg8=0.015, k=300, f=50):
    # subsample the matrix to make computation faster
    mat = mat[0:mat.shape[0]//128, 0:mat.shape[1]//128]
    mat = mat[mat.getnnz(1)>0][:, mat.getnnz(0)>0]

    print(mat.shape)
    no_users = mat.shape[0]
    no_movies = mat.shape[1]

    #baseline_bu, baseline_bi = baseline_estimator(mat)
    # We should call baseline_estimator but we can init at random for test
    baseline_bu, baseline_bi = np.random.rand(no_users, 1)  * 2 - 1, np.random.rand(1, no_movies) * 2 - 1    

    bu_index, bi_index = pre_processing(mat, mat_file)
    
    # Init parameters
    bu = np.random.rand(no_users, 1)  * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    wij = np.random.rand(no_movies, no_movies) * 2 - 1
    cij = np.random.rand(no_movies, no_movies) * 2 - 1
    qi = np.random.rand(no_movies, f) * 2 - 1
    pu = np.random.rand(no_users, f) * 2 - 1
    yj = np.random.rand(no_movies, f) * 2 - 1

    mu = mat.data[:].mean()
    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)

    # Train
    print("Train...")
    n_iter = 200
    cx = mat.tocoo()
    for it in range(n_iter):
        for u,i,v in zip(cx.row, cx.col, cx.data):
            #Rk_iu = Nk_iu = bi_index[u]
            N_u = bi_index[u]
            Rk_iu = Nk_iu = np.flip(np.argsort(S[i,]))[:k]
            e_ui = compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj)

            bu[u] += gamma1 * (e_ui - l_reg6 * bu[u])
            bi[0, i] += gamma1 * (e_ui - l_reg6 * bi[0, i])
            qi[i] += gamma2 * (e_ui * (pu[u] + 1 / sqrt(len(N_u)) * yj[N_u].sum(0)) - l_reg7 * qi[i])
            pu[u] += gamma2 * (e_ui * qi[i] - l_reg7 * pu[u])
            for j in N_u:
                yj[j] += gamma2 * (e_ui * 1/ sqrt(len(N_u)) * qi[i] - l_reg7 * yj[j])
            for j in Rk_iu:
                buj = mu + baseline_bu[u] + baseline_bi[0, j]
                wij[i][j] += gamma3 * ( 1 / sqrt(len(Rk_iu)) * e_ui * (mat[u, j] - buj) - l_reg8 * wij[i][j] )
                cij[i][j] += gamma3 * ( 1 / sqrt(len(Nk_iu)) * e_ui - l_reg8 * cij[i][j] )
        gamma1 *= 0.9
        gamma2 *= 0.9
        gamma3 *= 0.9

        if it % 10 == 0:
          print(it, "\ ", n_iter)         
          print("compute loss...")
          print(compute_loss(mat, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, qi, pu, N_u, yj, l_reg6=l_reg6, l_reg7=l_reg7, l_reg8=l_reg8))
#################################################


if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']
    integrated_model(mat, mat_file)
