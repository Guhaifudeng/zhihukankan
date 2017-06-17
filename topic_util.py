#encoding:utf-8
import codecs
import numpy as np
def build_topic_index_hasmap(topic_keys_file, has_head = False,index = False):
    tk_read = codecs.open(topic_keys_file, 'r', 'utf-8')
    line = tk_read.readline()
    tk_list = line.strip().split('\t')
    t_map = {}
    ind = 0
    for tk in tk_list:
        t_map[tk] = ind
        ind += int(index)
    print('load %s finished' % topic_keys_file)
    return t_map
