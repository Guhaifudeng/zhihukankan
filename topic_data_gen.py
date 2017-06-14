#encoding:utf-8
import codecs

def gen_topic_key(topic_file,topic_key_file,has_head = False):
    topic = set()
    t_write = codecs.open(topic_key_file,'w', 'utf-8')
    with codecs.open(topic_file, 'r', 'utf-8') as t_read:
        count = 0
        t_key = ''
        while True:
            line = t_read.readline()
            if not line:
                #print("00")
                break
            if has_head:
                has_head = False
                continue
            count += 1
            if count % 100 == 0:
                print('load topic count %d' % count)
            tp_list = line.strip().split('\t')

            topic.add(tp_list[0])
            tp_str = tp_list[1]
            if tp_list[0] != 'c':
                tp_2 = tp_list[1].strip().split(',')
                for tp in tp_2:
                    topic.add(tp)

        #print(t_key)
        t_write.write('\t'.join(tp))
        
        print("count of topic in topic_info %d" % len(topic))
    print("finished !")
    print(ind)
if __name__ == '__main__':
    gen_topic_key('../data/topic_info.txt','../out/topic_keys.txt',True)
