#encoding:utf-8
import pickle
if __name__ == "__main__":
    mlknn_in = open('../out/ml_knn.pkl','rb')
    ph_has_t = pickle.load(mlknn_in) #0
    ph_has_no_t = pickle.load(mlknn_in) #1
    has_ti_gma_p = pickle.load(mlknn_in) #2
    has_no_ti_gma_p = pickle.load(mlknn_in) #3
    mlknn_in.close()
    print('load ml_knn pickle finished')
    print(ph_has_t.shape)
    print(ph_has_no_t.shape)
    print(has_ti_gma_p.shape)
    print(has_no_ti_gma_p.shape)
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

    question_eval_embedding_file = '../out/question_eval_phrase_set.txt'
    question_eval_prediction = '../out/question_eval_prediction.txt'
    predition_write = codecs.open(question_eval_prediction,'w','utf-8')
    with codecs.open(question_eval_embedding_file,'r','utf-8') as eval_read:
        while True:
            line = eval_read.readline()
            print(line)
            if not line:
                print('read %s finished !' % question_eval_embedding_file)
                break
            q_eval_id,q_eval_vec = line.split('\t')
            q_eval_vec = q_eval_vec.split(',')
            dist,ind = tree.query([q_eval_vec],k = 2001)
            #是否包含ti
            q_eval_ti= np.add.reduce(phrase_topic_mat[ind,:],axis = 0)
            print(q_eval_ti.shape)
            out_has_or_not_0_EH = [tmp2[tmp1]for tmp1,tmp2 in zip(q_eval_ti,has_no_ti_gma_p)]
            out_has_or_not_1_EH =[tmp2[tmp1]for tmp1,tmp2 in zip(q_eval_ti,has_ti_gma_p)]
            print(out_has_or_not_0.shape)
            out_has_or_not_0 = np.multiply(out_has_or_not_0_EH,ph_has_no_t)
            out_has_or_not_1 = np.multiply(out_has_or_not_1_EH,ph_has_t)
            out_has_or_not_flag = out_has_or_not_0 < out_has_or_not_1
            #包含情况下ti发生的概率顺序
            out_has_or_not_01 = out_has_or_not_0 + out_has_or_not_1
            out_ti_p = np.divide(out_has_or_not_1,out_has_or_not_01)
            out_ti_p_sorted = np.argsort(-out_ti_p)
            print(out_ti_p_sorted.shape)
            end_ti = []
            for ti_p in out_ti_p_sorted:
                if out_has_or_not_flag[ti_p]:
                    tmp = topic_ind_key[ti_p]
                    end_ti.append(tmp)
            predition_write.write(','.join(end_ti)+'\n')

