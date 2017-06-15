#encoding:utf-8
import codecs
import os
import random
def extract_question_wordseq(quesiton_train_file, out_question_file):
    quesitons = []

    q_write = codecs.open(out_question_file,'w','utf-8')
    with codecs.open(quesiton_train_file,'r','utf-8') as file_read:
        count = 0
        while True:

            line = file_read.readline()
            #print(line)
            if not line:
                break

            count += 1
            if count % 10000 == 0:
                print("extract question count: %d" % count)
            q_id,q_c,q_w,note_c,note_w = line.split('\t')
            q_write.write(q_id + '\t' + q_w +'\t'+note_w)
        print("line count: %d" % count)


def random_extract_question(question_file,extract_question_file,extract_question_count = 10000,training_question_count = 2999967):
    quesitons = []
    min_rand_int = 0
    max_rand_int = 0
    if training_question_count > extract_question_count:
        max_rand_int = training_question_count // extract_question_count
    rand_ind = random.randint(min_rand_int, max_rand_int)
    extract_ind = 0
    q_write = codecs.open(extract_question_file,'w','utf-8')
    with codecs.open(question_file,'r','utf-8') as file_read:
        count = 0

        while True:
            line = file_read.readline()
            #print(line)
            if not line:
                break

            if count % 100 == 0:
                print("extract question count: %d" % count)
            q_id,q_c,q_w,note_c,note_w = line.split('\t')
            if extract_ind == rand_ind:
                count += 1
                q_write.write(q_id + '\t' + q_w +'\t'+note_w+'\n')
            if count >= extract_question_count:
                break
            if extract_ind >= max_rand_int:
                extract_ind = 0
            extract_ind += 1

        print("line count: %d" % count)
    print("finished!")

if __name__ == '__main__':
    #extract_question_wordembedding('H:/python35code/zhihutopic/data/question_train_set.txt',r'../out/question_word_extract.txt')
    random_extract_question('../data/question_train_set.txt','../out/random_10000_question.txt',extract_question_count= 40000)

