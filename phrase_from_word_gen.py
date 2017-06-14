#encoding:utf-8
import codecs
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer

    # question_file = '../data/question_train_set.txt'
    # question_topic_file = '../data/question_topic_train_set.txt'
    # word_keys_file = '../out/word_keys.txt'
    # topic_keys_file ='../out/topic_keys.txt'
    # out_tfidf_global_file ='../out/global_tfidf.txt'
    # out_idf_global_file = '../out/global_idf.txt'
    # gen_topic_word_tfidf(question_file,question_topic_file,\
    #     word_keys_file,topic_keys_file,out_tfidf_global_file,out_idf_global_file)
def remove_inf(i):
    if i == float("inf"):
        return 0
    return i

def gen_topic_word_tfidf(question_file,question_topic_file,
        word_keys_file,topic_keys_file,out_tfidf_global_file,out_idf_global_file):
    question_read = codecs.open(question_file,'r','utf-8')
    question_topic_read = codecs.open(question_topic_file,'r','utf-8')
    word_keys_read = codecs.open(word_keys_file,'r','utf-8')
    topic_keys_read = codecs.open(topic_keys_file,'r','utf-8')
    tfidf_global_write = codecs.open(out_tfidf_global_file,'w','utf-8')
    idf_write = codecs.open(out_idf_global_file,'w','utf-8')
    set_word = {}
    set_topic = {}
    dict_question_topic = {}
    #setting
    np.set_printoptions(threshold='nan')

    #no head
    line = word_keys_read.readline()
    line_list = line.strip().split('\t')
    ind = 0
    for key in line_list:
        set_word[key] = ind
        ind += 1
    print("load word key finished")

    #no head
    line = topic_keys_read.readline()
    line_list = line.strip().split('\t')
    ind = 0
    for key in line_list:
        set_topic[key] = ind
        #print(1)
        ind += 1
    #print(ind)
    print("load topic key finished")

    #no head
    while True:
        line = question_topic_read.readline()
        if not line:
            print('read q_t finished !')
            break
        q_id, t_s = line.split('\t')
        t_arr = t_s.strip().split(',')
        dict_question_topic[q_id] = t_arr

    print(len(dict_question_topic))
    print('load question-topic pair finished')
    print(len(set_topic),len(set_word))
    #no head
    freq_word = np.zeros(len(set_word))

    count = 0
    t_w_arr = np.zeros((len(set_topic),len(set_word)),dtype= bool)
    print(t_w_arr.dtype)
    while True:
        line = question_read.readline()
        if not line:
            print('read q finished !')
            print('count is %d' % count)
            break
        count += 1
        if count % 10000 ==0:
            print('dealt question : %d' % count)
        q_id,tm,q_vec,tm,n_vec = line.split('\t')
        #wrong wrong
        q_word_list = q_vec.strip().split(',')
        #n_word_list = n_vec.split('\t')

        t_arr = dict_question_topic[q_id]
        if len(t_arr) >8:
            print("id %d" % len(t_arr))
        for word in q_word_list:

            if word in set_word:
                w_ind = set_word[word]
                freq_word[w_ind] += 1
            else:
                print("error")
                print(word)
                continue
            #print(t_arr)
            for topic in t_arr:
                t_ind = set_topic[topic]
                t_w_arr[t_ind][w_ind] = 1
    #print(freq_word[4:80])
    #print(t_w_arr[4:80,1])
    print("load question-train finished")
    print("compute the bag of words finished")
    #idf
    idf_write.write('\t'.join(set_word)+'\n')
    tfidf_global_write.write('\t'.join(set_word)+'\n')

    idf = np.sum(t_w_arr,axis=0)
    print("idf-sum")
    print(idf.dtype)
    print(idf[1:200])
    idf = np.log(np.divide(len(set_topic)*1.0,idf))
    idf = np.array(map(lambda x: remove_inf(x),idf))
    print("idf-log")
    print(idf.dtype)
    print(idf[1:200])

    idf_write.write('\t'.join(map(lambda x:str(x),idf))+'\n')
    idf_write.flush()
    idf_write.close()
    print("output the idf of words finished")


    #compute TF-IDF
    # print(idf)
    #freq
    tfidf = np.multiply(freq_word, idf)
    print("tfidf")
    print(tfidf.dtype)
    print(tfidf[1:200])

    tfidf_global_write.write('\t'.join(map(lambda x:str(x),tfidf))+'\n')
    tfidf_global_write.flush()
    tfidf_global_write.close()
    print("output the tf-idf of words finished")
    # print('tfidf finished !')
    # tmp = ''
    # for i in list(set_topic): #topic
    #     tmp += '\t'+ i
    # idf_w.write(tmp[1:])
    # tfidf_w.write(tmp[1:])
    # tmp = ''
    # for i in list(set_word):#word
    #     tmp += '\t' + i
    # idf_w.write(tmp[1:])
    # tfidf_w.write(tmp[1:])

    # tmp = ''
    # for i in range(len(set_topic)):
    #     tmp = ''
    #     for j in range(len(set_word)):
    #         count += 1
    #         if count % 10000:
    #             print('dealt question : %d' % count)
    #         tmp += '\t'+ str(tfidf_arr[i,j])
    #     tfidf_w.write(tmp[1:])
    # print("compute the tf-idf of words finished")

if __name__ == "__main__":
    question_file = '../data/question_train_set.txt'
    question_topic_file = '../data/question_topic_train_set.txt'
    word_keys_file = '../out/word_keys.txt'
    topic_keys_file ='../out/topic_keys.txt'
    out_tfidf_global_file ='../out/global_tfidf.txt'
    out_idf_global_file = '../out/global_idf.txt'
    gen_topic_word_tfidf(question_file,question_topic_file,\
        word_keys_file,topic_keys_file,out_tfidf_global_file,out_idf_global_file)
