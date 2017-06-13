#encoding:utf-8

import codecs

w_f = codecs.open('/home/xiefeng/zhihu/topic_key.txt','w','utf-8')
dict = set()
count = 0
with codecs.open('/home/xiefeng/zhihu/topic_info.txt','r','utf-8') as f:
	
	while True:
		line = f.readline()
		if not line:
			print('finished !')
			break
			
		id = line.strip().split('\t' and ' ')[0]
		#print(id)
		dict.add(id)

		count += 1
		if count % 10000 == 0:
			print(count)
tmp = ''
print('---33-')
count = 0
for key in list(dict):

	count += 1
	if count % 10000 == 0:
		print('key %d' % count)
	tmp += '\t' + key
#print(tmp)
print('---22-')
w_f.write(tmp[1:])
print('---11-')
