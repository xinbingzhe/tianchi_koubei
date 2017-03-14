import datetime
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_pay.txt"
way1 = 'r'
road2 = "d:/tianchi_koubei/mid_dataset/count_shop_pay2.txt"
way2 = 'w'

def dateRange(start, end):
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(i), "%Y-%m-%d") for i in range(days)]
def initial(datelist,dic):
    for x in datelist:
        dic[x] = 0
datelist = dateRange("2015-07-01","2016-10-31")

fr = open(road1,way1)
fw = open(road2,way2)
line = fr.readline()
shop_dic={}
i = 0
print("begin")
while i<100000:
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
    line = fr.readline()
    i+=1
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
fr.close()
fw.close()
print('success')
  