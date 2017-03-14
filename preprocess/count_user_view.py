import datetime
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_view.txt"
way1 = 'r'
road3 = "d:/tianchi_koubei/origin_dataset/extra_user_view/20170112/extra_user_view.txt"
road2 = "d:/tianchi_koubei/mid_dataset/count_user_view.txt"
way2 = 'w'

def dateRange(start, end):
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(i), "%Y-%m-%d") for i in range(days)]
def initial(datelist,dic):
    for x in datelist:
        dic[x] = 0
datelist = dateRange("2015-07-01","2016-10-31")

fr1 = open(road1,way1)
fr2 = open(road3,way1)
fw = open(road2,way2)
line = fr1.readline()


shop_dic={}
i = 0
print("begin")
while line:
    re= line.strip('\n').split(',')
    shopid = re[1]
    day = re[2].split(' ')[0]
    #print(day)
    if shopid not in shop_dic:
        shop_dic[shopid]={}
        
        initial(datelist,shop_dic[shopid])
        shop_dic[shopid][day] += 1
    else:
        if day not in shop_dic[shopid]:
            shop_dic[shopid][day] = 1
        else:
            shop_dic[shopid][day] += 1
    line = fr1.readline()
    #i+=1
print("second file")
line2 = fr2.readline()
j = 0
while line2:
    re= line2.strip('\n').split(',')
    shopid = re[1]
    day = re[2].split(' ')[0]
    #print(day)
    if shopid not in shop_dic:
        shop_dic[shopid]={}
        
        initial(datelist,shop_dic[shopid])
        shop_dic[shopid][day] += 1
    else:
        if day not in shop_dic[shopid]:
            shop_dic[shopid][day] = 1
        else:
            shop_dic[shopid][day] += 1
    line2 = fr2.readline()
    #j+=1

print("output")
for shopid in shop_dic:
    output = str(shopid)
    for day in datelist:
        sum_liuliang = shop_dic[shopid][day]
        output +=','+str(sum_liuliang)
    output +='\n'
    #print(output)
    fw.write(output)
#print(shop_dic)

fr1.close()
fr2.close()
fw.close()
print('success')
  