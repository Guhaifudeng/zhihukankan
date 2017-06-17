#encoding:utf-8
import word_util
import numpy as np
import codecs
def transform_wordseq_to_phrase_weighted(word_seq,word2vec_map,word_weighted_value = None,word_keys = None):
    phrase_distributed = np.zeros(256)
    for word in word_seq:
        #print("0")
        if not word_keys:
            if word not in word_keys:
                continue
        #print("1")

        if not word_weighted_value:
            phrase_distributed += word2vec_map[word]
        else:
            if word not in word_weighted_value:
                #print(word)
                continue
            #print(word2vec_map[word])
            #print(word_weighted_value[])
            #print(word2vec_map[word])
            #print(word_weighted_value[word])
            weight = word_weighted_value[word]
            phrase_distributed += [word2vec_elem*weight for  word2vec_elem in word2vec_map[word]]
        #print('2')
    return phrase_distributed
def build_questions_vector_hashmap(phrase_embedding_file,question_count,has_head = False):
    dict_prase_vec = {}
    with codecs.open(phrase_embedding_file, 'r', 'utf-8') as p_read:
        count  = 0
        while True:
            line = p_read.readline()
            if not line:
                print('load %s finised' % phrase_embedding_file)
                break
            if has_head:
                pass
                has_head = False
                continue
            count += 1
            if count % 1000 == 0:
                print('load train sample %s' % count)
            phrase_id, phrase_vec= line.split('\t')
            phrase_vec = [float(i) for i in phrase_vec.split(',')]
            dict_prase_vec[phrase_id] = phrase_vec
            if count >= question_count:
                break
    print(count)
    return dict_prase_vec
def bulid_question_topic_hashmap(question_topic_file, has_head = False):
    dict_question_topic = {}
    with codecs.open(question_topic_file,'r', 'utf-8') as question_topic_read:
        #no head
        while True:
            line = question_topic_read.readline()
            if not line:
                print('read q_t finished !')
                break
            q_id, t_s = line.split('\t')
            t_arr = t_s.strip().split(',')
            dict_question_topic[q_id] = t_arr
    print('load %s finished' % question_topic_file)
    return dict_question_topic


if __name__ == "__main__":
    question_40000_file = '../out/random_40000_question.txt'
    question_40000_phrase_distributed_file = '../out/random_40000_question_embedding.txt'

    #question_train_file = '../data/question_train_set.txt'
    #question_train_phrase_vector_file = '../out/question_train_phrase_set.txt'
    question_eval_file =  '../data/question_eval_set.txt'
    question_eval_phrase_vector_file = '../out/question_eval_phrase_set.txt'

    word_embedding_file = '../data/word_embedding.txt'
    word2vec_map = word_util.build_word2vec_hashmap(word_embedding_file,has_head=True)
    word_tfidf_file = '../out/global_tfidf.txt'
    word_weighted_tfidf = word_util.build_word_tfidf_hashmap(word_tfidf_file)
    word_keys_file = '../out/word_keys.txt'
    word_keys = word_util.build_word_keys_hashmap(word_keys_file)

    p_write = codecs.open(question_40000_phrase_distributed_file, 'w', 'utf-8')
    #eval_write = codecs.open(filename)
    #train_write = codecs.open(question_train_phrase_vector_file, 'w','utf-8')
    eval_write = codecs.open(question_eval_phrase_vector_file, 'w', 'utf-8')

    count = 0
    with codecs.open(question_40000_file, 'r', 'utf-8') as train_read:
        while True:
            line = train_read.readline()
            if not line:
                print("read %s finised! " % question_40000_phrase_distributed_file)
                break
            q_id,q_w_seq,c_w_seq = line.split('\t')
            #print(q_id)
            #print(q_w_seq)
            q_w_seq = q_w_seq.split(',')
            #print(c_w_seq)
            q_w = transform_wordseq_to_phrase_weighted(q_w_seq, word2vec_map,word_weighted_tfidf,word_keys)
            #print(q_w)
            q_w = [str(e) for e in q_w.tolist()]
            p_write.write(q_id +'\t' + ','.join(q_w)+'\n')
            count += 1
            if count % 10000 == 0:
                print('train transform count: %d' % count)
    print('train set finised')


    # count = 0
    # with codecs.open(question_train_file, 'r', 'utf-8') as train_read:
    #     while True:
    #         line = train_read.readline()
    #         if not line:
    #             print("read %s finised! " % question_train_file)
    #             break
    #         q_id,_,q_w_seq,_,c_w_seq = line.split('\t')
    #         #print(q_id)
    #         #print(q_w_seq)
    #         q_w_seq = q_w_seq.split(',')
    #         #print(c_w_seq)
    #         q_w = transform_wordseq_to_phrase_weighted(q_w_seq, word2vec_map,word_weighted_tfidf,word_keys)
    #         #print(q_w)
    #         q_w = [str(e) for e in q_w.tolist()]
    #         train_write.write(q_id +'\t' + ','.join(q_w)+'\n')
    #         count += 1
    #         if count % 10000 == 0:
    #             print('train transform count: %d' % count)
    # print('train set finised')
    count = 0
    with codecs.open(question_eval_file, 'r', 'utf-8') as eval_read:
        while True:
            line = eval_read.readline()
            if not line:
                print("read %s finised! " % question_eval_file)
                break
            q_id,_,q_w_seq,_,c_w_seq = line.split('\t')
            #print(q_id)
            #print(q_w_seq)
            q_w_seq = q_w_seq.split(',')
            #print(c_w_seq)
            q_w = transform_wordseq_to_phrase_weighted(q_w_seq, word2vec_map,word_weighted_tfidf,word_keys)
            #print(q_w)
            q_w = [str(e) for e in q_w.tolist()]
            eval_write.write(q_id +'\t' + ','.join(q_w)+'\n')
            count +=1
            if count % 10000 == 0:
                print('eval transform count: %d' % count)
    print('eval set finised')
