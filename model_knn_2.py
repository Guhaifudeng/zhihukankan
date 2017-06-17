#encoding:utf-8
import numpy as np
import sklearn
import question_util
import topic_util
import codecs
import pickle
from sklearn.neighbors import KDTree
if __name__ == "__main__":
    tree_in = open('../out/kd_tree.pkl','rb')
    topic_ind_key = pickle.load(tree_in) #0
    topic_key_ind = pickle.load(tree_in) #1
    phrase_key_ind = pickle.load(tree_in) #2
    phrase_ind_key = pickle.load(tree_in) #3
    phrase_vec = pickle.load(tree_in) #4
    point_topic_mat = pickle.load(tree_in) #5
    dist = pickle.load(tree_in) #6
    ind = pickle.load(tree_in) #7
    tree = pickle.load(tree_in)#8
    tree_in.close()
    print(dist[0])
    print(ind[0])
