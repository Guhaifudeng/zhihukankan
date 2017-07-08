#encoding:utf-8
import numpy as np
import sklearn
import question_util
import topic_util
import codecs
import pickle
from sklearn.neighbors import KDTree
def build_question_topic_matric_01(question_topic_file,point_topic_mat,phrase_key_ind,topic_key_ind):
    with codecs.open(question_topic_file,'r','utf-8') as qt_read:
        while True:
            line = qt_read.readline()
            if not line:
                break
            q_id, t_s = line.split('\t')
            t_arr = t_s.strip().split(',')
            if q_id not in phrase_key_ind:
                continue
            for t_e in t_arr:
                point_topic_mat[phrase_key_ind[q_id],topic_key_ind[t_e]] = True
    print("load %s finished" % question_topic_file)
    return point_topic_mat
if __name__ == '__main__':

    #用空间换时间
    phrase_embedding_file = '../out2/random_40000_question_embedding.txt'
    question_count = 40000
    phrase_hashmap= question_util.build_questions_vector_hashmap(phrase_embedding_file, question_count)
    phrase_ind_key = np.array(list(phrase_hashmap.keys()))
    phrase_vec = np.array(list(phrase_hashmap.values()))
    ind = 0
    phrase_key_ind = {}
    for key in phrase_ind_key:
        phrase_key_ind[key] = ind
        ind += 1
    phrase_hashmap.clear()

    #用空间换时间
    topic_keys_file = '../out2/topic_keys.txt'
    topic_key_ind = topic_util.build_topic_index_hasmap(topic_keys_file,index= True)
    topic_ind_key = {}
    ind = 0
    for key in topic_key_ind:
        topic_ind_key[ind] = key
        ind += 1

    #布尔型矩阵 40000*2000
    question_topic_file = '../data/question_topic_train_set.txt'
    point_topic_mat = np.zeros((len(phrase_key_ind),len(topic_key_ind)),dtype= bool)
    point_topic_mat = build_question_topic_matric_01(question_topic_file, point_topic_mat, phrase_key_ind, topic_key_ind)


    #print(phrase_ind_key[0])
    tree_out = open('../out2/kd_tree.pkl', 'wb')
    tree = KDTree(phrase_vec, leaf_size= int(1.5* len(phrase_key_ind)))
    #s =
    #tree_copy = pickle.loads(s)
    dist, ind = tree.query(phrase_vec, k=4001)
    print(dist[0],ind[0])

    #save data with pickle
    pickle.dump(topic_ind_key, tree_out,-1) #0
    pickle.dump(topic_key_ind, tree_out,-1) #1
    pickle.dump(phrase_key_ind, tree_out, -1) #2
    pickle.dump(phrase_ind_key, tree_out, -1) #3
    pickle.dump(phrase_vec, tree_out,-1) #4
    pickle.dump(point_topic_mat, tree_out, -1) #5
    pickle.dump(dist, tree_out,-1) #6
    pickle.dump(ind, tree_out,-1) #7
    pickle.dump(tree,tree_out,-1) #8
    tree_out.close()

    tree_in = open('../out2/kd_tree.pkl','rb')
    for i in range(6):
        pickle.load(tree_in)
    dist = pickle.load(tree_in)
    ind = pickle.load(tree_in)
    tree_in.close()
    print(dist[0])
    print(ind[0])

