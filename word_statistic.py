#encoding:utf-8
import word_util
import codecs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
def word_idf_equal_zero(word_idf_map):
    
    idf_0_list = list(map(lambda x:float(word_idf_map[x])<1e-6,word_idf_map))
    #print(type(idf_0_list))
    count_idf_count_zero = np.sum(idf_0_list)
    #print(type(count_idf_count_zero))

    return count_idf_count_zero,count_idf_count_zero/np.size(idf_0_list)

def plot_hist_word_idf(x):
    fig= plt.subplots(1, 1, figsize = (8, 4))

    plt.hist(x)
    plt.show()
if __name__ == "__main__":
    word_idf_file = "../out/global_idf.txt"
    word_idf_map = word_util.build_word_idf_hashmap(word_idf_file)
    count_idf_0,proportion_idf_0 = word_idf_equal_zero(word_idf_map)
    print(count_idf_0, proportion_idf_0)
    list_idf = list(map(lambda x:float(word_idf_map[x]),word_idf_map))
    #print(list_idf)
    plot_hist_word_idf(list_idf)
