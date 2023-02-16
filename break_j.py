import jieba

documents = ['我来自北京清华大学', '我喜欢写程式', '每天发技术文章']
# 精確模式
for sentence in documents:
    seg_list = jieba.cut(sentence)
    print('/'.join(seg_list))