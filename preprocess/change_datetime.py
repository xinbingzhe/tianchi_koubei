import datetime
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_pay.txt"
way1 = 'r'
road2 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_pay_day.txt"
way2 = 'w'
fr = open(road1,way1)
fw = open(road2,way2)
line = fr.readline()
i = 0
print("begin")
while line:
    re= line.strip('\n').split(' ')
    fw.write(re[0]+'\n')
    #print(re[0]+'\n')
    line = fr.readline()
    i += 1

fr.close()
fw.close()
print('success')