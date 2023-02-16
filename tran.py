f_before = open('Red_1.txt', 'r', encoding = 'utf-8')
f_after = open('Red_1_after.txt', 'w+', encoding = 'utf-8')

words = []
for line in f_before:
	words.append(line)



for word in range(len(words)):
	UpperQuote = 0
	LowerQuote = 0
	while UpperQuote != -1:
		UpperQuote = words[word].find('「', UpperQuote+1)
		LowerQuote = words[word].find('」', UpperQuote+1)	
		#print(UpperQuote)
		#print(LowerQuote)
		#print(words[word][:UpperQuote])
		if UpperQuote != -1:
			words[word] = words[word][:UpperQuote] + '\r\n' + words[word][LowerQuote+1:]
	f_after.write(words[word])
# for text in words:
# 	f_after.write(words[text])

f_before.close()
f_after.close()

