import re
f = open("words.txt")
text = f.read()
print(type(text))
f.close()
# for i in text:
# 	re.sub("\'", "", i)
f = open("res.txt", 'w')

res = text.strip('][').split(', ')
print(type(res[0]))
for i in range(len(res)):
	res[i] = re.sub("\'", "", res[i])

print(type(res))
print(res, file=f)
f.close()
