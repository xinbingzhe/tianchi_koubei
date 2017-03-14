print('begin')
import numpy as np
import matplotlib.pyplot as plt 
road1 = "E:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct_3.0.txt"
way2 = 'w'
fr = open(road1,way1)
fw = open(road2,way2)

def create_xday(xday):
    for i in range(1,490):
        xday.append(i)
    return xday
xday_list = []
xday = create_xday(xday_list)
def create_weekend(xweekend):
    j = 3
    for i in range(1,490):
        if j == 6 or j ==7:
            xweekend.append(1)
            if j == 7:
                j = 1
            else:
                j += 1
        else:
            xweekend.append(0)
            j += 1
    return xweekend
xweekend_list = []
xweekend = create_weekend(xweekend_list)


def readfile_oneshop_Y(fr):
    line = fr.readline()
    re = line.strip('\n').split(',')
    data_str = map(float,re[1:])
    data_float = []
    for s in data_str:
        data_float.append(s)
    return data_float
### 计算需要填补的值，分开周末数据和平时的数据
def output(fw,shopid,data):
    data_str = []
    data_int = map(int,data)
    data_tostr = map(str,data_int)
    for i in data_tostr:
        data_str.append(i)
    fw.write(str(shopid)+','+','.join(data_str)+'\n')
def compute(num,data,xday,xweekend):
    #print('#')
    normalday_data = []
    weekend_data = []
    if num < 14:
        end = num + 14
        #print(end)
        #print(xday[:53])
        for n,w,d in zip(xday[:end],xweekend[:end],data[:end]):
            if w == 0:
                normalday_data.append(d)
            else:
                weekend_data.append(d)
    elif 14 <=num < 336:
        for n,w,d in zip(xday[num-14:num],xweekend[num-14:num],data[num-14:num+14]):
            if w == 0:
                normalday_data.append(d)
            else:
                weekend_data.append(d)
    elif num >= 336:
        for n,w,d in zip(xday[num-14:],xweekend[num-14:],data[num-14:]):
            if w == 0:
                normalday_data.append(d)
            else:
                weekend_data.append(d)
    if xweekend[num] == 0:
        return np.mean(normalday_data)

    else:
        return np.mean(weekend_data)
i = 0
while i<2000:
    pay_day_total = readfile_oneshop_Y(fr)
    pay_first = pay_day_total[:-350]
    pay_day = pay_day_total[-350:]
    for d in range(0,len(pay_day)):
        if pay_day[d]==0 or 0<pay_day[d]<8:
            if d < 14:
                if pay_day[d] < 15*np.mean(pay_day[:14]):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])
                
            elif 14 <= d < 336:
                if pay_day[d] < 15*np.mean(pay_day[d-14:d+14]):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])
                
            elif d>= 336:
                if pay_day[d] < 15*np.mean(pay_day[d-14:]):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])
        else:
            if d < 14:
                if pay_day[d] > 8*np.mean(pay_day[:14]):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])
            elif 14 <= d < 336:
                if pay_day[d] > 2*(np.mean(pay_day[d-14:d])+np.mean(pay_day[d:d+14])):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])
            elif d>= 336:
                if pay_day[d] > 4*np.mean(pay_day[d-14:]):
                    pay_day[d] = compute(d,pay_day,xday[-350:],xweekend[-350:])


    correct_pay = pay_first + pay_day
    #print(len(correct_pay))
    output(fw,i+1,correct_pay)

    print(i+1)
    i += 1
fr.close()
fw.close()
