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
            if not line:
                print("load word2vec finished")
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
if __name__ == '__main__':
    print(build_word2vec_hashmap('../data/word_embedding.txt',has_head= True).popitem())
