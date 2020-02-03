from scipy import io, sparse
import numpy as np
from itertools import groupby
from operator import itemgetter
import pickle
import os
from scipy import io, sparse
import numpy as np
from utils import pre_processing



#################################################
def compute_loss(mat, mu, bu, bi, l_reg=0.02):
  loss = 0

  no_users_entries = np.array((mat != 0).sum(1)).T.ravel()
  bu_rep = np.repeat(bu.ravel(), no_users_entries)

  no_movies_entries = np.array((mat != 0).sum(0)).ravel()
  bi_rep = np.repeat(bi.ravel(), no_movies_entries)

  temp_mat = sparse.csc_matrix(mat).copy()
  temp_mat.data[:] -= bi_rep
  temp_mat.data[:] -= mu
  temp_mat = sparse.coo_matrix(temp_mat)
  temp_mat = sparse.csr_matrix(temp_mat)
  temp_mat.data[:] -= bu_rep

  loss = (temp_mat.data[:] ** 2).sum()

  reg = l_reg * ((bu**2).sum() + (bi**2).sum())  
  loss += reg

  return loss


def baseline_estimator(mat, mat_file, l_reg=0.02, learning_rate=0.0000025):
  """mat = mat[1:5000,1:5000]
  mat = mat[mat.getnnz(1)>0][:,mat.getnnz(0)>0]"""

  print(mat.shape)
  no_users = mat.shape[0]
  no_movies = mat.shape[1]
  
  bu_index, bi_index = pre_processing(mat, mat_file)

  bu = np.random.rand(no_users,1)  * 2 - 1
  bi = np.random.rand(1,no_movies) * 2 - 1
  #bu = np.zeros((no_users,1))
  #bi = np.zeros((1,no_movies))  

  mu = mat.data[:].mean()
  mat_sum1 = mat.sum(1)
  mat_sum0 = mat.sum(0)
  n = mat.data[:].shape[0]

  no_users_entries = np.array((mat != 0).sum(1))
  no_movies_entries = np.array((mat != 0).sum(0))

  # Train
  print("Train...")
  n_iter = 200
  for it in range(n_iter):

    #bi_sum = bi[bi_index].sum(1).reshape((no_users,1))
    #bu_sum = bu.ravel()[bu_index].sum(0).reshape((1,no_movies)) 

    bi_sum = np.array(list(map(lambda x:bi.ravel()[x].sum(), bi_index))).reshape((no_users,1))
    bu_sum = np.array(list(map(lambda x:bu.ravel()[x].sum(), bu_index))).reshape((1,no_movies))    

    # Vectorized operations
    bu_gradient = - 2.0 * (mat_sum1 - no_users_entries  * mu - no_users_entries  * bu - bi_sum) + 2.0 * l_reg * bu
    bu -= learning_rate * bu_gradient 

    bi_gradient = - 2.0 * (mat_sum0 - no_movies_entries * mu - no_movies_entries * bi - bu_sum) + 2.0 * l_reg * bi
    bi -= learning_rate * bi_gradient 
 
    if it % 10 == 0:
      print(it, "\ ", n_iter)         
      print("compute loss...")
      print(compute_loss(mat, mu, bu, bi, l_reg=l_reg))

  return bu, bi
#################################################



if __name__ == "__main__":
    mat_file = path+"/T.mat"
    mat = io.loadmat(mat_file)['X']    
    baseline_estimator(mat, mat_file)
