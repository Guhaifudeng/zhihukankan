#encoding:utf-8

import codecs
import word_util
def gen_word_key(word_embedding_file,word_key_file,has_head = False):

    w_write = codecs.open(word_key_file,'w', 'utf-8')
    with codecs.open(word_embedding_file, 'r', 'utf-8') as w_read:
        count = 0
        w_key = ''
        while True:
            line = w_read.readline()
            if not line:
                break
            if has_head:
                has_head = False
                continue
            count  +=  1
            if count % 1000 == 0:
                print('load word count %d' % count)
            w_key += '\t' + line.strip().split(" ")[0]
            #print(w_key)
        w_write.write(w_key[1:]+'\n')
        print("count of words in word_embedding_file %d" % count)
    print("finished !")

def gen_word_key_after_removed(word_idf_map, word_key_after_removed_file):
    with codecs.open(word_key_after_removed_file,'w','utf-8') as w_write:
        rm_list = []
        for (key, idf) in word_idf_map.items():
            if float(idf) < 1e-6:
                rm_list.append(key)
        for key in rm_list:
            word_idf_map.pop(key)
        word_key = word_idf_map.keys()
        w_write.write('\t'.join(word_key)+'\n')
        w_write.close()
def gen_word_tfidf_after_removed(word_keys_tfidf_after_removed_file,word_tfidf_map,word_keys):
    with codecs.open(word_keys_tfidf_after_removed_file,'w','utf-8') as w_tfidf_write:
        word_keys = []
        word_tfidf = []
        for (key, tfidf) in word_tfidf_map.items():
            if key  in word_keys:
                word_keys.append(key)
                word_tfidf.append(tfidf)
        w_tfidf_write.write('\t'.join(word_keys)+'\n')
        w_tfidf_write.write('\t'.join(word_tfidf)+'\n')
        w_tfidf_write.close()

if __name__ == '__main__':
    #gen_word_key('../data/word_embedding.txt','../out/word_keys.txt',True)
    word_key_tfidf_after_removed_file = '../out/partition_tfidf.txt'
    # word_idf_map = word_util.build_word_idf_hashmap('../out/global_idf.txt')
    # gen_word_key_after_removed(word_idf_map, word_key_after_removed_file)
    word_tfidf_map = word_util.build_word_tfidf_hashmap('../out/global_tfidf.txt')
    word_keys = word_util.build_word_keys_hashmap('../out/word_keys_rmd.txt')
    gen_word_tfidf_after_removed(word_key_tfidf_after_removed_file,word_tfidf_map,word_keys)    
    print('finished')
