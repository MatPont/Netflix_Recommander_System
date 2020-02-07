import utils

import sys
import os
from scipy.sparse import dok_matrix, csr_matrix
from scipy import io
import tarfile



#################################################
total_no_users = 2649429
total_no_movies = 17770

def process_content(content, D):
    lines = content.split("\n")
    id_movie = int(lines[0][:-1]) - 1
    for i in range(1, len(lines)):
        if lines[i] != '':
            line = lines[i].split(",")
            id_user = int(line[0]) - 1
            rating = int(line[1])
            D[id_user, id_movie] = rating
    return D


def rating_compiler(folder_name, out_path):
    D = dok_matrix((total_no_users, total_no_movies))
    res_listdir = os.listdir(folder_name)
    number = len(res_listdir)
    i = 0
    for f in res_listdir:
        if os.path.isfile(folder_name+f):
            print(i, " / ", number)
            myfile = open(folder_name+f)
            content = myfile.read()
            myfile.close()
            D = process_content(content, D)
        i += 1
    D = csr_matrix(D)             
    io.savemat(out_path, {'X' : D})


def rating_compiler2(tar_name, out_path):
    D = dok_matrix((total_no_users, total_no_movies))
    tar = tarfile.open(tar_name)
    res_getmembers = tar.getmembers()
    number = len(res_getmembers)
    i = 0
    for member in res_getmembers:
        f = tar.extractfile(member)
        if f is not None:    
            print(i, " / ", number)        
            content = f.read()
            f.close()
            D = process_content(content.decode(), D)
        i += 1
    tar.close()
    D = csr_matrix(D)             
    io.savemat(out_path, {'X' : D})


def extract_T_and_R(D_file_name, file_name, out_T_path, out_R_path):
    D = io.loadmat(D_file_name)['X']
    myfile = open(file_name)
    content = myfile.read()
    myfile.close()
    lines = content.split("\n")
    users, movies = set(), set()
    for line in lines:
        if line != '':
            line_split = line.split(":")
            if len(line_split) == 2:
                # Movie id
                movies.add(int(line_split[0]) - 1)
            else:
                # User id
                users.add(int(line_split[0]) - 1)
    T = D[list(users),:]
    T = T[:,list(movies)]    
    io.savemat(out_T_path, {'X' : T})
    
    movies2 = set(range(total_no_movies))
    movies2 = movies2.difference(movies)
    users2 = set(range(total_no_users))
    users2 = users2.difference(users)
    
    R = D[list(users2),:]
    R = R[:,list(movies2)]
    io.savemat(out_R_path, {'X' : R})
    



#################################################
if __name__ == "__main__":
    #rating_compiler2(path+"/download/training_set.tar", path+"/D.mat")
    #extract_T_and_R(path+"/D.mat", path+"/download/qualifying.txt", path+"/T.mat", path+"/R.mat")
    extract_T_and_R(path+"/D.mat", path+"/download/probe.txt", path+"/T.mat", path+"/R.mat")
