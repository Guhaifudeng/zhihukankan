#encoding:utf-8

# import codecs
# from sklearn.feature_extraction.text import TfidfTransformer
# import numpy as np
# a = np.zeros((200,400),dtype= bool)
# a[4][5] = True
# print(a.dtype)
# # #doukeyi
# tmp = np.sum(a,axis=1)
# print(tmp.dtype)
# print(tmp)
# def remove_inf(i):
#     if i == float("inf"):
#         return 0
#     return i
# idf = np.log(np.divide(50*1.0,tmp))
# idf = np.array(map(lambda x:remove_inf(x),idf))
# print(idf.dtype)
# print(idf)
# np.set_printoptions(threshold='nan')
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import mlab
# from matplotlib import rcParams
# fig1 = plt.figure(2)
# rects =plt.bar(left = (0.2,1),height = (1,0.5),width = 0.2,align="center",yerr=0.000001)
# plt.title('Pe')
# plt.show()
# hello = 'sdfs\thhs\tsfsds'
# _,_,a = hello.split('\t')
# print(a)
# a = [1.00000000,1.232323]
# a = [str(e) for e in a]
# print(','.join(a))


import numpy as np
# np.random.seed(0)
# X = np.random.random((10, 3))  # 10 points in 3 dimensions
# tree = KDTree(X, leaf_size=2)
# dist, ind = tree.query([X[0]], k=3)
# print(ind)  # indices of 3 closest neighbors
# print(dist)  # distances to 3 closest neighbors
a = np.random.randint(0,5,5)
b = np.random.randint(0,9,5)
print(a)
print(b)
d = np.argsort(-b)
print(d)
print(b[d])
c = a > b
print(c)
print(a[c])
