#encoding:utf-8

import codecs
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
        w_write.write(w_key[1:])
        print("count of words in word_embedding_file %d" % count)
    print("finished !")
if __name__ == '__main__':
    gen_word_key('../data/word_embedding.txt','../out/word_keys.txt',True)
