path = "../Datasets"

from itertools import groupby
from operator import itemgetter
import pickle
import os
from sklearn.preprocessing import StandardScaler

#path = "/content/drive/My Drive/M2/AFMatriciel"

def compute_sparse_correlation_matrix(A):
    scaler = StandardScaler(with_mean=False)
    scaled_A = scaler.fit_transform(A)  # Assuming A is a CSR or CSC matrix
    corr_matrix = (1/scaled_A.shape[0]) * (scaled_A.T @ scaled_A)
    return corr_matrix

def pre_processing(mat, mat_file):
    # Create bu and bi indexes
    # bi_index is a list with a size equal to the number of users
    #    the jth element is a list storing the indexes of movies rated by user j
    # bu_index is the same but storing the indexes of users whose rating is 
    #    available for a movie
    # These indexes will help to vectorize computation of the gradient

    shape = str(mat.shape[0])+"_"+str(mat.shape[1])
    bu_index_file = mat_file+"_bu_index_"+shape+".data"
    bi_index_file = mat_file+"_bi_index_"+shape+".data"

    if not (os.path.isfile(bu_index_file) and os.path.isfile(bi_index_file)):
        #mat = io.loadmat(mat_file)['X']
        """mat = mat[1:5000,1:5000]
        mat = mat[mat.getnnz(1)>0][:,mat.getnnz(0)>0]"""

        print("Pre-processing...")
        mat_nonzero = mat.nonzero()
        """cx = mat.tocoo()    
        bi_index = [[]]*mat.shape[0]
        bu_index = [[]]*mat.shape[1]
        for i,j,v in zip(cx.row, cx.col, cx.data):
          bi_index[i].append(j)
          bu_index[j].append(i)
        print(bi_index[0])"""

        print("   make bi indexes...")
        bi_index = []
        for k, g in groupby(zip(mat_nonzero[0], mat_nonzero[1]), itemgetter(0)):
          to_add = list(map(lambda x:int(x[1]), list(g)))
          bi_index.append(to_add)

        print("   make bu indexes...")
        bu_index = []
        indexes = np.argsort(mat_nonzero[1])
        for k, g in groupby(zip(mat_nonzero[1][indexes], mat_nonzero[0][indexes]), itemgetter(0)):
          to_add = list(map(lambda x:int(x[1]), list(g)))
          bu_index.append(to_add)    

        with open(bi_index_file, "wb") as fp:
            pickle.dump(bi_index, fp)
        with open(bu_index_file, "wb") as fp:
            pickle.dump(bu_index, fp)
    else:
        with open(bi_index_file, "rb") as fp:
            bi_index = pickle.load(fp)
        with open(bu_index_file, "rb") as fp:
            bu_index = pickle.load(fp)

    print("Pre-processing done.")
    return bu_index, bi_index
