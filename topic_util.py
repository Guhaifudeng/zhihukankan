#encoding:utf-8
import codecs
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def topic_statistic(question_topic_pair_file,dict_topic):
    q_read = codecs.open(question_topic_pair_file,'r','utf-8')
    while True:
        line = q_read.readline()
        if not line:
            print('read q finished !')
            break
    #print(line)
        q_id,t_s = line.split('\t')
        t_arr = t_s.strip().split(',')
        for t in t_arr:
            dict_topic[t] += 1

    return list(map(lambda x:dict_topic[x],dict_topic))
def build_topic_map(topic_keys_file):
    tk_read = codecs.open(topic_keys_file, 'r', 'utf-8')
    line = tk_read.readline()
    tk_list = line.split('\t')
    t_map = {}
    for tk in tk_list:
        t_map[tk] = 0
    return t_map

def plot_hist_topic_count(x):

    fig = plt.figure()
    plt.bar(range(len(x)),x,0.4,color="green")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("topic hist")
if __name__ == '__main__':
    quetsion_topic_pair_file = '../data/question_topic_train_set.txt'
    topic_keys_file = '../out/topic_keys.txt'
    print(3)
    t_map = build_topic_map(topic_keys_file)
    print("2")
    t_count_list =topic_statistic(quetsion_topic_pair_file, t_map)
    print("1")
   #codecs.open('../out/topic_hist.txt','w','utf-8').write(t_count_list)
    plot_hist_topic_count(t_count_list)
    print(4)
