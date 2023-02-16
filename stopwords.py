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
stopwords = ['你', '道', '來', '說', '了', '也', '又', '是', '的', '我', ' ', '']
jieba.load_userdict("RMdic.txt")

for line in f: 
	temp = jieba.lcut(line)
	words = []
	#print(temp)
	for i in temp:
		#i = ''.join(i)
		#print(i)
		i = re.sub("[\.\!\/_,$%^*(+\"\'””《》]+|[+——！。？，、~@#￥%……&*（）：+「」『』\n\u3000你道我的是又也來了說]", "", i)
		if len(i) > 0:
			words.append(i)
	if len(words) > 0:
		lines.append(words)
	


# list1 = []
# del_i = []
# del_j = []
# for index_i in range(len(lines)):
# 	for index_j in range(len(lines[index_i])):
# 		for i in range(len(stopwords)):
# 			if lines[index_i][index_j] == stopwords[i]:
# 				del_i.append(index_i)
# 				del_j.append(index_j)
# print(len(del_i))
# print(len(del_j))
# # print(lines[del_i[84327]][del_j[84327]])

# for i in range(0, 84327):
# 	del lines[del_i[i]][del_j[i]]

f = open("temp.txt", 'w')
print(lines, file=f)
f.close()
# f = open("word.txt", 'w')
# print(list1, file=f)
# f.close()

# word_to_idx = {}
# idx_to_word = {}
# ids = 0

# for w in list1:
# 	cnt = word_to_idx.get(w, [ids, 0])
# 	if cnt[1] == 0:
# 		ids += 1
# 	cnt[1] += 1
# 	word_to_idx[w] = cnt
# 	idx_to_word[ids] = w
# print(word_to_idx["我"])
# print(type(word_to_idx["甄士隱"]))


# f=  open("sorted.txt", 'w')
# for i in word_to_idx:
# 	if word_to_idx[i][1] > 3945:
# 		print(i, file = f)
# f.close()

# f =  open("word_to_idx.txt", 'w') 
# #print(len(words))
# print(word_to_idx, file=f)
# f.close()

# f =  open("idx_to_word.txt", 'w') 
# #print(len(words))
# print(idx_to_word, file=f)
# f.close()