#encoding:utf-8
import codecs
def word_tokenize_str_to_float(words_string):
    #other operations
    words_list = [float(i) for i in words_string.strip().split(',') if i !=""]
    return words_list

def build_word2vec_hashmap(word_embedding_file, has_head = False):

    dict_word_vec = {}
    with codecs.open(word_embedding_file, 'r', 'utf-8') as w_read:
        count = 0
        while True:
            line = w_read.readline()
            if not line:
                print("load %s finished" % word_embedding_file)
                break
            if has_head:
                word_count, word_embedding_dimension = line.strip().split(' ')
                has_head = False
                continue
            count += 1
            if count % 10000 == 0:
                print('load word count %d' % count)
            '''
            print(count)
            #w_id,w_vec_str = line.strip().split('\t' or ' ')
            w_vec_float =  word_tokenize(w_vec_str)
            if len(w_vec_float) != 256:
                print('word embeding is bad !')
                break
            dict_word_vec[w_id] = w_vec_float
            '''
            id_vec = line.strip().split('\t' and ' ')
            if len(id_vec) != 257:
                print('word embedding is bad !')
                break
            dict_word_vec[id_vec[0]] = [float(i) for i in id_vec[1:]]
    #test= list(map(lambda x:word_tokenize(x[1]),dict_word_vec.items()))
    print(count)
    return dict_word_vec


def build_word_idf_hashmap(word_idf_file, has_head = False):
    dict_word_idf = {}
    with codecs.open(word_idf_file,'r','utf-8') as idf_read:
        line_keys = idf_read.readline().strip().split('\t')
        line_idf = idf_read.readline().strip().split('\t')
        for key,idf in zip(line_keys, line_idf):
            dict_word_idf[key] = float(idf)
    print('word %s finished' % word_idf_file)
    return dict_word_idf


def build_word_tfidf_hashmap(word_tfidf_file, has_head = False):
    dict_word_tfidf = {}
    with codecs.open(word_tfidf_file,'r','utf-8') as tfidf_read:
        line_keys = tfidf_read.readline().strip().split('\t')
        line_tfidf = tfidf_read.readline().strip().split('\t')
        for key,tfidf in zip(line_keys, line_tfidf):
            dict_word_tfidf[key] = float(tfidf)
    print('word %s finished !'% word_tfidf_file)
    return dict_word_tfidf

def build_word_keys_hashmap(word_keys_file,has_head = False,index = False):
    with codecs.open(word_keys_file,'r','utf-8') as word_keys_read:
        line = word_keys_read.readline()
        line_list = line.strip().split('\t')
        set_word = {}
        ind = 0
        for key in line_list:
            set_word[key] = ind
            ind += int(index)
        print("load %s finished" % word_keys_file)
    return build_word_keys_hashmap

if __name__ == '__main__':
    # print(build_word2vec_hashmap('../data/word_embedding.txt',has_head= True).popitem())
    print(build_word_idf_hashmap('../out/global_idf.txt'))
