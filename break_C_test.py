#-*- conding:utf-8 -*-
import re 
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ckiptagger import data_utils, construct_dictionary, WS

#數值運算包
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#sklearn
from sklearn.decomposition import PCA

#word2vec
import gensim as gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.word2vec import LineSentence

with open("RM2.txt", encoding = 'utf-8') as f: 
	text = f.read()
#print(text)
f.close()

word_to_weight = {
    "賈演": 3,
    "賈寅": 3, 
    "賈源": 3, 
    "賈法": 3, 
    "賈代化": 3, 
    "賈代善": 3, 
    "賈代儒": 3, 
    "賈代修": 3, 
    "賈敷": 3, 
    "賈敬": 3, 
    "賈赦": 3, 
    "賈政": 3, 
    "賈敏": 3, 
    "賈敕": 3, 
    "賈效": 3, 
    "賈敦": 3, 
    "賈珍": 3, 
    "賈璉": 3, 
    "賈琮": 3, 
    "賈珠": 3, 
    "賈寶玉": 3, 
    "寶玉": 3, 
    "賈環": 3, 
    "賈瑞": 3, 
    "賈璜": 3, 
    "賈珩": 3, 
    "賈㻞": 3, 
    "賈珖": 3, 
    "賈琛": 3, 
    "賈瓊": 3, 
    "賈璘": 3,
    "賈元春": 3, 
    "賈迎春": 3, 
    "賈探春": 3, 
    "賈惜春": 3, 
    "喜鸞": 3, 
    "四姐": 3, 
    "賈蓉": 3, 
    "賈蘭": 3, 
    "賈薔": 3, 
    "賈菌": 3, 
    "賈芸": 3, 
    "賈芹": 3, 
    "賈萍": 3, 
    "賈菖": 3, 
    "賈菱": 3, 
    "賈蓁": 3, 
    "賈藻": 3, 
    "賈蘅": 3, 
    "賈芬": 3, 
    "賈芳": 3, 
    "賈芝": 3, 
    "賈荇": 3, 
    "賈芷": 3, 
    "賈葛": 3, 
    "賈巧姐": 3, 
    "史太君": 3, 
    "史鼐": 3, 
    "史鼎": 3, 
    "史湘雲": 3, 
    "王子騰": 3,
    "王子勝": 3, 
    "王夫人": 3, 
    "薛姨媽": 3, 
    "王仁": 3, 
    "王熙鳳": 3, 
    "薛蟠": 3, 
    "薛蝌": 3, 
    "薛寶釵": 3, 
    "寶釵": 3, 
    "薛寶琴": 3, 
    "林黛玉": 3, 
    "妙玉": 3, 
    "邢夫人": 3, 
    "尤氏": 3, 
    "李紈": 3, 
    "秦可卿": 3, 
    "胡氏": 3, 
    "許氏": 3, 
    "香菱": 3, 
    "趙姨娘": 3, 
    "劉姥姥": 3, 
    "甄寶玉": 3, 
    "襲人": 3, 
    "珍珠": 3, 
    "媚人": 3, 
    "秋紋": 3, 
    "晴雯": 3, 
    "綺霰": 3, 
    "綺霞": 3, 
    "麝月": 3, 
    "檀雲": 3, 
    "碧浪": 3, 
    "碧痕": 3, 
    "茜雪": 3, 
    "春燕": 3, 
    "墜兒": 3, 
    "四兒": 3, 
    "小燕": 3, 
    "蕙香": 3,
    "佳蕙": 3, 
    "抱琴": 3, 
    "司棋": 3, 
    "蓮花兒": 3, 
    "繡橘": 3, 
    "待書": 3, 
    "翠墨": 3, 
    "蟬姐": 3, 
    "入畫": 3, 
    "彩屏": 3, 
    "紫鵑": 3, 
    "雪雁": 3, 
    "春纖": 3, 
    "鴛鴦": 3, 
    "琥珀": 3, 
    "玻璃": 3, 
    "翡翠": 3, 
    "鸚鵡": 3, 
    "靛兒": 3, 
    "傻大姐": 3, 
    "銀蝶": 3, 
    "炒豆兒": 3, 
    "卐兒": 3, 
    "鶯兒": 3, 
    "文杏": 3, 
    "平兒": 3, 
    "小紅": 3, 
    "豐兒": 3, 
    "金釧": 3,
    "玉釧": 3, 
    "繡鸞": 3, 
    "繡鳳": 3, 
    "彩雲": 3, 
    "彩霞": 3, 
    "素雲": 3, 
    "同喜": 3, 
    "同貴": 3, 
    "縷兒": 3, 
    "翠縷": 3, 
    "寶珠": 3, 
    "侍書": 3, 
    "瑞珠": 3, 
    "姣杏": 3, 
    "小螺": 3, 
    "善姐": 3, 
    "臻兒": 3, 
    "篆兒": 3, 
    "定兒": 3, 
    "小吉祥兒": 3, 
    "小鵲": 3, 
    "小舍兒": 3, 
    "寶蟾": 3, 
    "茗煙": 3, 
    "焙茗": 3, 
    "焦大": 3, 
    "李貴": 3, 
    "鋤藥": 3, 
    "墨雨": 3, 
    "伴鶴": 3, 
    "掃花": 3, 
    "引泉": 3, 
    "挑芸": 3, 
    "雙瑞": 3, 
    "雙壽": 3, 
    "來旺": 3, 
    "興兒": 3, 
    "王榮": 3, 
    "錢啟": 3, 
    "張若錦": 3, 
    "趙亦華": 3, 
    "錢槐": 3, 
    "小玄兒": 3, 
    "隆兒": 3, 
    "昭兒": 3, 
    "喜兒": 3, 
    "住兒": 3, 
    "壽兒": 3, 
    "杏奴": 3, 
    "慶兒": 3, 
    "王信": 3, 
    "芳官": 3, 
    "齡官": 3, 
    "蕊官": 3, 
    "藕官": 3, 
    "荳官": 3, 
    "寶官": 3, 
    "文官": 3, 
    "茄官": 3, 
    "菂官": 3, 
    "艾官": 3, 
    "玉官": 3, 
    "葵官": 3, 
    "頑石": 3, 
    "茫茫大士": 3, 
    "癩頭僧人": 3, 
    "渺渺真人": 3, 
    "跛足道人": 3, 
    "空空道人": 3, 
    "情僧": 3, 
    "甄士隱": 3, 
    "封氏": 3, 
    "小童": 3, 
    "神瑛侍者": 3, 
    "絳珠仙子": 3, 
    "警幻仙子": 3, 
    "賈雨村": 3, 
    "雨村": 3, 
    "嚴老爺": 3, 
    "霍啟": 3, 
    "封肅": 3, 
    "冷子興": 3, 
    "林如海": 3, 
    "李嬤嬤": 3, 
    "王嬤嬤": 3, 
    "門子": 3, 
    "李守中": 3, 
    "馮淵": 3, 
    "拐子": 3, 
    "痴夢仙姑": 3, 
    "引愁金女": 3, 
    "鍾情大士": 3, 
    "度恨菩提": 3, 
    "王成": 3, 
    "劉氏": 3, 
    "板兒": 3, 
    "青兒": 3, 
    "周瑞": 3, 
    "智能": 3, 
    "余信": 3, 
    "秦鍾": 3, 
    "賴二": 3, 
    "詹光": 3, 
    "戴良": 3, 
    "錢華": 3, 
    "單聘仁": 3, 
    "吳新登": 3, 
    "秦業": 3, 
    "胡氏": 3, 
    "金氏": 3, 
    "馮唐": 3, 
    "張友士": 3, 
    "戴權": 3, 
    "牛清": 3, 
    "牛繼宗": 3, 
    "柳彪": 3, 
    "柳芳": 3, 
    "陳翼": 3, 
    "陳瑞文": 3, 
    "馬魁": 3, 
    "馬尚": 3, 
    "侯曉明": 3, 
    "侯孝康": 3, 
    "石光珠": 3, 
    "蔣子寧": 3, 
    "謝鯨": 3, 
    "戚建輝": 3, 
    "裘良": 3, 
    "馮紫英": 3, 
    "陳也俊": 3, 
    "衛若蘭": 3, 
    "水溶": 3, 
    "二丫頭": 3, 
    "淨虛": 3, 
    "智善": 3, 
    "胡老爺": 3, 
    "金哥": 3, 
    "李公子": 3, 
    "雲光": 3, 
    "夏守忠": 3, 
    "賴大": 3, 
    "趙嬤嬤": 3, 
    "吳天佑": 3, 
    "吳貴妃": 3, 
    "卜固修": 3, 
    "山子野": 3, 
    "林之孝": 3, 
    "程日興": 3, 
    "昭容": 3, 
    "彩繽": 3, 
    "花母": 3, 
    "花自芳": 3, 
    "多官": 3, 
    "王嫂子": 3, 
    "周氏": 3, 
    "卜世仁": 3, 
    "銀姐": 3, 
    "倪二": 3, 
    "王短腿": 3, 
    "方椿": 3, 
    "馬道婆": 3, 
    "周姨娘": 3, 
    "胡斯來": 3, 
    "鮑太醫": 3, 
    "王濟仁": 3, 
    "蔣玉菡": 3, 
    "雲兒": 3, 
    "張道士": 3, 
    "周奶娘": 3, 
    "傅試": 3, 
    "傅秋芳": 3, 
    "宋嬤嬤": 3, 
    "茗玉": 3, 
    "王君效": 3, 
    "賴大的母親": 3, 
    "金彩": 3, 
    "金文翔": 3, 
    "嫣紅": 3, 
    "柳湘蓮": 3, 
    "冷郎君": 3, 
    "賴尚榮": 3, 
    "邢岫煙": 3, 
    "邢忠": 3, 
    "李嬸娘": 3, 
    "李紋": 3, 
    "李綺": 3, 
    "梅翰林": 3, 
    "胡君榮": 3, 
    "良兒": 3, 
    "烏進孝": 3, 
    "婁氏": 3, 
    "女先兒": 3, 
    "單大良": 3, 
    "趙國基": 3, 
    "單大娘": 3, 
    "祝媽": 3, 
    "田媽": 3, 
    "葉媽": 3, 
    "許氏": 3, 
    "何婆子": 3, 
    "小鳩兒": 3, 
    "夏婆子": 3, 
    "佩鳳": 3, 
    "柳五兒": 3, 
    "偕鸞": 3, 
    "尤二姐": 3, 
    "尤三姐": 3, 
    "尤老娘": 3, 
    "張華": 3, 
    "俞祿": 3, 
    "秋桐": 3, 
    "天文生": 3, 
    "潘又安": 3, 
    "朱大娘": 3, 
    "周太監": 3, 
    "小霞": 3, 
    "翠雲": 3, 
    "張媽": 3, 
    "邢德全": 3, 
    "文花": 3, 
    "圓信": 3, 
    "智通": 3, 
    "孫紹祖": 3, 
    "夏金桂": 3, 
    "夏奶奶": 3, 
    "王一貼": 3,
}
dictionary = construct_dictionary(word_to_weight)
#print(dictionary)

#load dictionary
ws = WS("./data")
#break the sentence to word and store in list
ws_results = ws([text])
temp = []
words = []

# two-dimentional to one-dimentional
for index_i in range(len(ws_results)):
	for index_j in range(len(ws_results[index_i])):
		temp.append(ws_results[index_i][index_j])


for i in temp:
	#replace all punctuation to ""
	i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：]+", "", i)
	if len(i) > 0:
		words.append(i)

#print(len(words))
#print(words)

trigrams = [([words[i], words[i + 1]], words[i + 2]) for i in range(len(words) - 2)]
print(trigrams[:3])


#get the list
vocab = set(words)
print(len(vocab))

word_to_idx = {}
idx_to_word = {}
ids = 0

for w in words:
	cnt = word_to_idx.get(w, [ids, 0])
	if cnt[1] == 0:
		ids += 1
	cnt[1] += 1
	word_to_idx[w] = cnt
	idx_to_word[ids] = w
print(word_to_idx["甄士隱"])


class NGram(nn.Module):

	def __init__(self, vocab_size, embedding_dim, context_size): 
		super(NGram, self).__init__()
		self.embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.linear1 = nn.Linear(context_size * embedding_dim, 128)
		self.linear2 = nn.Linear(128, vocab_size)

	def forward(self, inputs):
		embeds = self.embeddings(inputs).view(1, -1)
		#ReLU
		out = F.relu(self.linear1(embeds))

		#Softmax
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

	def extract(self, inputs):
		embeds = self.embeddings(inputs)
		return embeds

#loss fuction
losses = []
criterion = nn.NLLLoss()
model = NGram(len(vocab), 10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.001)

with open("loss.txt", encoding = 'utf-8') as f: 
	for epoch in range(100):
		total_loss = torch.Tensor([0])
		for context, target in trigrams:
			context_idxs = [word_to_idx[w][0] for w in context]

			context_var = Variable(torch.LongTensor(context_idxs))

			optimizer.zero_grad()

			log_probs = model(context_var)

			loss = criterion(log_probs, Variable(torch.LongTensor([word_to_idx[target][0]])))

			loss.backward()

			optimizer.step()

			total_loss += loss.data
		losses.append(total_loss)
		print('第{}輪，損失函數：{:.2f}\n'.format(epoch, total_loss.numpy()[0]), file=f)

f.close()
with open("vec.txt", encoding = 'utf-8') as f: 
	vec = model.extract(Variable(torch.LongTensor([v[0] for v in word_to_idx.values()])))
	vec = vec.data.numpy()
	f.write(vec)
f.close()

with open("X_reduced.txt", encoding = 'utf-8') as f: 
	X_reduced = PCA(n_components = 2).fit_transform(vec)
	f.write(X_reduced)

fig = plt.figure(figsize = (30, 20))
ax = fig.gca()
ax.set_facecolor('black')
ax.plot(X_reduced[:, 0], X_reduced[:, 1], '.', markersize=1, alpha=0.4, color='white')

words = ['賈寶玉', '林黛玉', '薛寶釵', '王熙鳳', '甄士隱', '賈雨村', '空空道人', '史太君', '史湘雲', '平兒', '賈母']

zhfont1 = matplotlib.font_manager.FontProperties(fname='./DFKai-SB.ttf', size = 16)
for w in words:
	if w in word_to_idx:
		ind = word_to_idx[w][0]
		xy = X_reduced[ind]
		plt.plot(xy[0], xy[1], '.', alpha=1, color='red')
		plt.text(xy[0], xy[1], '.', fontproperties=zhfont1, alpha=1, color='white')

plt.show()

