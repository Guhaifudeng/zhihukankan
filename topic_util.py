#encoding:utf-8
import codecs
import numpy as np  
import matplotlib.pyplot as plt  
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

	return list(map(lambda x:dict[x],dict_topic))
def build_topic_map(topic_keys_file):
	tk_read = codecs.open(topic_keys_file, 'r', 'utf-8')
	line = tk_read.readline()
	tk_list = line.split('\t')
	t_map = {}
	for tk in tk_list:
		t_map[tk] = 0
	return t_map

def plot_hist_topic_count(x):
    mu,sigma=100,15
    n,bins,patches=plt.hist(x,100,normed=1,facecolor='g',alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60,.025, r'$\mu=100,\ \sigma=15$')
    plt.axis([40,160,0,0.03])
    plt.grid(True)
if __name__ == '__main__':
	quetsion_topic_pair_file = '../data/question_topic_train_set.txt'
	topic_keys_file = '../out/topic_keys.txt'
	t_map = build_topic_map(topic_keys_file)
	t_count_list =topic_statistic(quetsion_topic_pair_file, t_map)
	plot_hist_topic_count(t_count_list)