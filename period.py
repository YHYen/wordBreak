f_before = open('Red_1_after.txt', 'r', encoding = 'utf-8')
f_after = open('Red_1_final.txt', 'w+', encoding = 'utf-8')

words = []
for line in f_before:
	words.append(line)

for word in range(len(words)):
	Period = 0
	while Period != -1 :
		Period = words[word].find('。', Period+1)
		print(Period)
		if Period != -1:
			if Period < len(words[word])-1:
				words[word] = words[word][:Period+1] + '\r\n' + words[word][Period+1:]
	f_after.write(words[word])


# for text in words:。
# 	f_after.write(words[text])

f_before.close()
f_after.close()

