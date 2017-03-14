print("begin")
import datetime
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/user_pay_day.txt"
way1 = 'r'
road2 = "d:/tianchi_koubei/mid_dataset/shop_OldNew_user_feature.txt"
way2 = 'w'

def dateRange(start, end):
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(i), "%Y-%m-%d") for i in range(days)]
def initial(datelist,dic):
    for x in datelist:
        dic[x] = [0,0]
datelist = dateRange("2015-07-01","2016-10-31")
fr = open(road1,way1)
fw = open(road2,way2)
line = fr.readline()

shop_dic = [[],{}]
initial(datelist,shop_dic[1])

lastshop = '#'
i = 0

while line:
    re = line.strip('\n').split(',')
    userid = re[0]
    shopid = re[1]
    day = re[2].split(' ')[0]
    if shopid!= lastshop:
        #output
        old = ""
        new = ""
        oldp = ""
        newp = ""

        for d in datelist:
            new += ','+str(shop_dic[1][d][0])
            old += ','+str(shop_dic[1][d][1])
            if shop_dic[1][d][0] == 0:
                newp += ','+str(0)
            else:
                newp += ','+str(round(float(shop_dic[1][d][0]/(shop_dic[1][d][0]+shop_dic[1][d][1])),2))
            if shop_dic[1][d][1] == 0:
                oldp += ','+str(0)
            else:
                oldp += ','+str(round(float(shop_dic[1][d][1]/(shop_dic[1][d][0]+shop_dic[1][d][1])),2))
        fw.write(lastshop+new+'\n')
        fw.write(lastshop+old+'\n')
        fw.write(lastshop+newp+"\n")
        fw.write(lastshop+oldp+'\n')
        #initial
        shop_dic = [[],{}]
        initial(datelist,shop_dic[1])
        if day in datelist:
            if userid not in shop_dic[0]:
                # 新客户消费记录
                shop_dic[0].append(userid)
                shop_dic[1][day][0] += 1
            else:
                #老客户消费记录
                shop_dic[1][day][1] += 1
    else:
        if day in datelist:
            if userid not in shop_dic[0]:
                shop_dic[0].append(userid)
                shop_dic[1][day][0] += 1
            else:
                shop_dic[1][day][1] += 1
    lastshop = shopid
    line = fr.readline()
    i += 1
#output
old = ""
new = ""
oldp = ""
newp = ""

for d in datelist:
    new += ','+str(shop_dic[1][d][0])
    old += ','+str(shop_dic[1][d][1])
    if shop_dic[1][d][0] == 0:
        newp += ','+str(0)
    else:
        newp += ','+str(round(float(shop_dic[1][d][0]/(shop_dic[1][d][0]+shop_dic[1][d][1])),2))
    if shop_dic[1][d][1] == 0:
        oldp += ','+str(0)
    else:
        oldp += ','+str(round(float(shop_dic[1][d][1]/(shop_dic[1][d][0]+shop_dic[1][d][1])),2))
fw.write(lastshop+new+'\n')
fw.write(lastshop+old+'\n')
fw.write(lastshop+newp+'\n')
fw.write(lastshop+oldp+'\n')
fr.close()
fw.close()
print('success')