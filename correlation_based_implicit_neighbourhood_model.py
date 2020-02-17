from utils import pre_processing, compute_sparse_correlation_matrix, path

import numpy as np
from scipy import io, sparse
from math import sqrt
from time import time



# Through all this code Rk_iu and Nk_iu are the same since implicit matrix is
#    made from the rating matrix without additional information (i.e. indexes of
#    non-zero elements are the same therefore neighbors too).


#################################################
# Non-vectorized way (iterate through each r_ui)
#################################################
def predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi):
    buj = mu + baseline_bu[u] + baseline_bi[0, Rk_iu]
    Rk_iu_sum = np.multiply((mat[u, Rk_iu] - buj), wij[i][Rk_iu]).sum()
    Nk_iu_sum = cij[i][Rk_iu].sum()
    return mu + bu[u] + bi[0, i] + Rk_iu_sum / sqrt(len(Rk_iu)) + Nk_iu_sum / sqrt(len(Nk_iu))

def compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi):
    return mat[u, i] - predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi)

def compute_loss(mat, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, l_reg=0.002):
    loss = 0
    loss_reg = 0
    cx = mat.tocoo()        
    for u,i,v in zip(cx.row, cx.col, cx.data):
        r_ui_pred = predict_r_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi)
        Rk_iu_sum = (wij[i][Rk_iu] ** 2).sum()
        Nk_iu_sum = (cij[i][Rk_iu] ** 2).sum()
        loss += (mat[u, i] - r_ui_pred) ** 2 
        loss_reg += l_reg * ((bu ** 2).sum() + (bi ** 2).sum() + Rk_iu_sum + Nk_iu_sum) 

    return loss, loss+loss_reg

def correlation_based_implicit_neighbourhood_model(mat, mat_file, l_reg=0.002, gamma=0.005, l_reg2=100.0, k=250):
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

    mu = mat.data[:].mean()

    # Compute similarity matrix
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
        t0 = time()
        for u,i,v in zip(cx.row, cx.col, cx.data):
            #Rk_iu = Nk_iu = bi_index[u]
            Rk_iu = Nk_iu = np.flip(np.argsort(S[i,].toarray()))[:k].ravel()
            e_ui = compute_e_ui(mat, u, i, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi)

            bu[u] += gamma * (e_ui - l_reg * bu[u])
            bi[0, i] += gamma * (e_ui - l_reg * bi[0, i])

            buj = mu + baseline_bu[u] + baseline_bi[0, Rk_iu]
            wij[i][Rk_iu] += gamma * ( 1 / sqrt(len(Rk_iu)) * e_ui * (mat[u, Rk_iu].toarray().ravel() - buj) - l_reg * wij[i][Rk_iu] )
            cij[i][Nk_iu] += gamma * ( 1 / sqrt(len(Nk_iu)) * e_ui - l_reg * cij[i][Nk_iu] )
        gamma *= 0.99

        if it % 10 == 0:
          t1 = time()
          print(it, "\ ", n_iter, "(%.2g sec)" % (t1 - t0))
          print("compute loss...")
          print(compute_loss(mat, mu, bu, bi, Rk_iu, wij, Nk_iu, cij, baseline_bu, baseline_bi, l_reg=l_reg))

    return bu, bi, wij, cij
#################################################


#################################################
# Vectorized way (in work)
# (Actually this version is faster but updates e_ui
# less frequently making it less accurate for the
# gradient descent)
#################################################
def compute_e_vectorized(mat, mu, bu, bi, Rk, wij, Nk, cij, baseline_bu, baseline_bi):
    # Rk and Nk are list of tuple (u, i, Rk_iu/Nk_iu)

    no_users_entries = np.array((mat != 0).sum(1)).T.ravel()
    bu_rep = np.repeat(bu.ravel(), no_users_entries)

    no_movies_entries = np.array((mat != 0).sum(0)).ravel()
    bi_rep = np.repeat(bi.ravel(), no_movies_entries)

    temp_mat = sparse.csc_matrix(mat).copy()
    temp_mat.data[:] -= mu
    temp_mat.data[:] -= bi_rep
    Rk_sum = np.array(list(map(lambda x : ( (mat[x[0], x[2]].toarray().ravel() \
                                           - (mu + baseline_bu[x[0]] + baseline_bi[0, x[2]])) \
                                           * wij[x[1]][x[2]] ).sum() / sqrt(len(x[2])), Rk)))
    temp_mat.data[:] -= Rk_sum
    Nk_sum = np.array(list(map(lambda x : cij[x[1]][x[2]].sum() / sqrt(len(x[2])), Nk)))
    temp_mat.data[:] -= Nk_sum
    temp_mat = sparse.coo_matrix(temp_mat)
    temp_mat = sparse.csr_matrix(temp_mat)
    temp_mat.data[:] -= bu_rep

    return temp_mat

def compute_loss_vectorized(mat, mu, bu, bi, Rk, wij, Nk, cij, baseline_bu, baseline_bi, l_reg=0.002):
    no_nonzero_element = np.array((mat != 0).sum())
    loss = (compute_e_vectorized(mat, mu, bu, bi, Rk, wij, Nk, cij, baseline_bu, baseline_bi).data[:] ** 2).sum()
    loss_reg = l_reg * np.array(list(map(lambda x : (cij[x[1]][x[2]] ** 2).sum(), Nk))).sum()
    loss_reg += l_reg * np.array(list(map(lambda x : (wij[x[1]][x[2]] ** 2).sum(), Rk))).sum()
    loss_reg += no_nonzero_element * l_reg * (bu ** 2).sum()
    loss_reg += no_nonzero_element * l_reg * (bi ** 2).sum()

    return loss, loss+loss_reg

def correlation_based_implicit_neighbourhood_model_vectorized(mat, mat_file, l_reg=0.002, gamma=0.005, l_reg2=100.0, k=250):
    gamma /= 100

    # subsample the matrix to make computation faster
    mat = mat[0:mat.shape[0]//128, 0:mat.shape[1]//128]
    mat = mat[mat.getnnz(1)>0][:, mat.getnnz(0)>0]

    print(mat.shape)
    no_users = mat.shape[0]
    no_movies = mat.shape[1]
    no_users_entries = np.array((mat != 0).sum(1))
    no_movies_entries = np.array((mat != 0).sum(0))    

    #baseline_bu, baseline_bi = baseline_estimator(mat)
    # We should call baseline_estimator but we can init at random for testing
    baseline_bu, baseline_bi = np.random.rand(no_users, 1)  * 2 - 1, np.random.rand(1, no_movies) * 2 - 1    

    bu_index, bi_index = pre_processing(mat, mat_file)
    
    bu = np.random.rand(no_users, 1)  * 2 - 1
    bi = np.random.rand(1, no_movies) * 2 - 1
    wij = np.random.rand(no_movies, no_movies) * 2 - 1
    cij = np.random.rand(no_movies, no_movies) * 2 - 1

    mu = mat.data[:].mean()

    # Compute similarity matrix
    N = sparse.csr_matrix(mat).copy()
    N.data[:] = 1
    S = sparse.csr_matrix.dot(N.T, N)
    S.data[:] = S.data[:] / (S.data[:] + l_reg2)
    S = S * compute_sparse_correlation_matrix(mat)

    Rk = []
    cx = mat.tocoo()
    for u,i,v in zip(cx.row, cx.col, cx.data):
        Rk.append((u, i, np.flip(np.argsort(S[i,].toarray()))[:k].ravel()))

    # Train
    print("Train...")
    n_iter = 200
    for it in range(n_iter):
        t0 = time() 

        e = compute_e_vectorized(mat, mu, bu, bi, Rk, wij, Rk, cij, baseline_bu, baseline_bi)
        # Vectorized operations
        bu += gamma * (e.sum(1) - no_users_entries * l_reg * bu)
        bi += gamma * (e.sum(0) - no_movies_entries * l_reg * bi)

        # TODO: vectorize the following
        for u, i, Rk_iu in Rk:
            Nk_iu = Rk_iu
            e_ui = e[u, i]
            buj = mu + baseline_bu[u] + baseline_bi[0, Rk_iu]
            wij[i][Rk_iu] += gamma * ( 1 / sqrt(len(Rk_iu)) * e_ui * (mat[u, Rk_iu].toarray().ravel() - buj) - l_reg * wij[i][Rk_iu] )
            cij[i][Nk_iu] += gamma * ( 1 / sqrt(len(Nk_iu)) * e_ui - l_reg * cij[i][Nk_iu] )
        gamma *= 0.99

        if it % 10 == 0:
          t1 = time()
          print(it, "\ ", n_iter, "(%.2g sec)" % (t1 - t0))       
          print("compute loss...")
          print(compute_loss_vectorized(mat, mu, bu, bi, Rk, wij, Rk, cij, baseline_bu, baseline_bi, l_reg=l_reg))  

    return bu, bi, wij, cij
#################################################


if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']
    correlation_based_implicit_neighbourhood_model(mat, mat_file)
    #correlation_based_implicit_neighbourhood_model_vectorized(mat, mat_file)
