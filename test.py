#encoding:utf-8

import codecs
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
a = np.ones((20,40))
tmp = np.sum(a>0,axis=1)
print(tmp)
idf = np.divide(tmp, 50.0)
print(idf)