#encoding:utf-8
import numpy as np
import sklearn
import question_util
import topic_util
import codecs
import pickle
from sklearn.neighbors import KDTree
def compute_priror_probabilities_pH_class2000(phrase_topic_mat,s=1):
    H_has_t = np.add.reduce(phrase_topic_mat,axis = 0)
    ph_has_t = (H_has_t + s)/(s*2 + phrase_topic_mat.shape[0])
    ph_has_no_t = 1- ph_has_t
    return ph_has_t,ph_has_no_t
def compute_posterior_probabilities_pEH_class2000_mul_k2001(phrase_topic_mat,phrase_ind_mat,s = 1):
    count = 0
    print(count)
    k = phrase_ind_mat.shape[1]
    class_num = phrase_topic_mat.shape[1]
    phrase_num = phrase_topic_mat.shape[0]
    phrase_k_ti_count = np.zeros((phrase_num,class_num))
    # for i in range(phrase_ind_B.shape[0]):
    #     for j in range(phrase_topic_A.shape[1]):
    #         phrase_ti_count_C[i][j] = np.add.reduce(phrase_topic_A[phrase_ind_B[i],j])
    #问题-最近k包含该主题数
    for i in range(phrase_num):
        ind = phrase_ind_mat[i]
        count += 1
        if count % 10000 == 0:
            print(count)
        phrase_k_ti_count[i] = (np.add.reduce(phrase_topic_mat[ind,:],axis=0))
    print('---ph_k_ti finished')
    #主题-包含该主题-最近k包含主题数
    has_ti_count_in_k =  np.zeros((class_num,k+1))
    #主题-不包含该主题-最近k包含主题数
    has_no_ti_count_in_k = has_ti_count_in_k.copy()

    for t_ind in range(class_num):
        m_has_ti_bool = phrase_topic_mat[:,t_ind]
        mk_has_ti_count = phrase_k_ti_count[:,t_ind]
        for ki in range(k+1):
            count += 1
            if count % 10000 == 0:
                print(count)
            has_ti_count_in_k[t_ind,ki]= np.add.reduce(np.logical_and(mk_has_ti_count == ki,m_has_ti_bool))
            has_no_ti_count_in_k[t_ind,ki] = np.add.reduce(np.logical_and(mk_has_ti_count == ki ,np.logical_not(m_has_ti_bool)))
    print('---ph_count_in_k finished')
    #print(has_ti_gma)
    #print(has_no_ti_gma)
    #主题-包含该主题-最近k包含主题数
    has_ti_count_in_k_p =  np.zeros((class_num,k+1))
    #主题-不包含该主题-最近k包含主题数
    has_no_ti_count_in_k_p = has_ti_count_in_k_p.copy()

    #点P的k近邻出现ti计gma次,出现gma次时，点P标注ti的概率
    for t_ind in range(class_num):
        count += 1
        if count % 10000 == 0:
            print(count)
        has_ti_count_in_k_p[t_ind,:] = (s+has_ti_count_in_k[t_ind,:])/(s*(k+1) + np.add.reduce(has_ti_count_in_k[t_ind,:]))
        has_no_ti_count_in_k_p[t_ind,:] = (s+has_no_ti_count_in_k[t_ind,:])/(s*(k+1) + np.add.reduce(has_no_ti_count_in_k[t_ind,:]))
    print('---all finished')
    return has_ti_count_in_k_p, has_no_ti_count_in_k_p
if __name__ == "__main__":
    tree_in = open('../out/kd_tree.pkl','rb')
    topic_ind_key = pickle.load(tree_in) #0
    topic_key_ind = pickle.load(tree_in) #1
    phrase_key_ind = pickle.load(tree_in) #2
    phrase_ind_key = pickle.load(tree_in) #3
    pickle.load(tree_in) #4 phrase_vec
    phrase_topic_mat = pickle.load(tree_in) #5
    pickle.load(tree_in) #6 phrase_dist
    phrase_ind_mat = pickle.load(tree_in) #7
    pickle.load(tree_in)#8 tree
    tree_in.close()
    print('load pickle finished!')

    mlknn_out = open('../out/ml_knn.pkl','wb')
    ph_has_t,ph_has_no_t = compute_priror_probabilities_pH_class2000(phrase_topic_mat)
    print('---ph finished')
    has_ti_gma_p, has_no_ti_gma_p = compute_posterior_probabilities_pEH_class2000_mul_k2001(phrase_topic_mat, phrase_ind_mat)
    pickle.dump(ph_has_t, mlknn_out,-1) #0
    pickle.dump(ph_has_no_t, mlknn_out,-1) #1
    pickle.dump(has_ti_gma_p, mlknn_out,-1) #2
    pickle.dump(has_no_ti_gma_p, mlknn_out,-1) #3
    mlknn_out.close()
    print('save pickle finished')
