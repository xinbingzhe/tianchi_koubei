
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_pay_day.txt" 
way1 = 'r'

fr = open(road1,way1)

line = fr.readline()

dic = {}
i = 0
lastshopid = ''
while line:
    re= line.strip('\n').split(',')
    shopid = re[1]
    if shopid != lastshopid:
        if shopid not in dic:
            dic[shopid] = 1
        else:
            dic[shopid]+= 1
            print("err:"+str(i))
            print(re)
    
    lastshopid = shopid
    line = fr.readline()
    i += 1
fr.close()
print("success")
