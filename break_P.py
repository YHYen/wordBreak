#-*- conding:utf-8 -*-
import jieba
import re 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from ckiptagger import data_utils, construct_dictionary, WS

#數值運算包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#sklearn
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
#word2vec
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

f = open("RM2.txt", encoding = 'utf-8')
f = f.read().split("。")
lines = []
#print(text)

jieba.load_userdict("RMdic.txt")

for line in f: 
	temp = jieba.lcut(line)
	words = []
	#print(temp)
	for i in temp:
		#i = ''.join(i)
		#print(i)
		i = re.sub("[\.\!\/_,$%^*(+\"\'””《》]+|[+——！。；？，、~@#￥%……&*（）：+「」『』\n\u3000你道我的是又也來了說]", "", i)
		if len(i) > 0:
			words.append(i)
	if len(words) > 0:
		lines.append(words)
	
f = open("temp.txt", 'w')
print(lines, file=f)
f.close()

# print(list(f))
# print(lines)
# print(words)
# f.close()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Word2Vec(lines, vector_size=20, window=2, min_count=0, sg=0, epochs=100)
#tensor = tensor.to(device)
#model.to(device)
renwu = model.wv.most_similar('林黛玉', topn = 20)
print(renwu)

rawWordVec = []
word2ind = {}
for i, w in enumerate(model.wv.index_to_key):
	rawWordVec.append(model.wv[w])
	word2ind[w] = i

rawWordVec = np.array(rawWordVec)
f= open("vec.txt", 'a') 
print(model.wv.index_to_key, file=f)
f.close()
X_reduced = PCA(n_components=2).fit_transform(rawWordVec)
f = open("X_reduced.txt", 'a')
print(X_reduced, file=f)
f.close()



fig = plt.figure(figsize = (15, 10))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize = 1, alpha = 0.3, color = 'white')



words = ['賈寶玉', '林黛玉', '薛寶釵', '王熙鳳', '甄士隱', '賈雨村', '空空道人', '史太君', '史湘雲', '平兒', '賈母']

zhfont1 = matplotlib.font_manager.FontProperties(fname='./DFKai-SB.ttf', size=16)
for w in words:
    if w in word2ind:
        ind = word2ind[w]
        xy = X_reduced[ind]
        plt.plot(xy[0], xy[1], '.', alpha =1, color = 'red')
        plt.text(xy[0], xy[1], w, fontproperties = zhfont1, alpha = 1, color = 'yellow')
plt.show()


