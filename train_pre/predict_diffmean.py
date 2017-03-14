print('begin')
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
road1 = "e:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "e:/tianchi_koubei/result/diffmean_result.csv"
way2 = 'w'
def create_xday(xday):
    for i in range(1,504):
        xday.append(i)
    return xday
xday_list = []
xday = create_xday(xday_list)
def create_weekend(xweekend):
    j = 3
    for i in range(1,504):
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
def mean_normal_weekend_diff(pay_day,xday,xweekend,start,end):
    y_pre = []
    nor_pay = []
    weekend_pay = []
    for p,n,w in zip(pay_day[start:end],xday[start:end],xweekend[start:end]):
        if w == 1:
            weekend_pay.append(p)
        else:
            nor_pay.append(p)
    nor = np.mean(nor_pay)
    weekend = np.mean(weekend_pay)
    for d in xweekend[end:]:
        if d == 0:
            y_pre.append(nor)
        else:
            y_pre.append(weekend)
    return y_pre
fr1 = open(road1,way1)
#fr2 = open(road2,way1)
fw = open(road3,way2)

i = 0
#读取特征
def readfile_oneshop_X(fr,xday,xweekend):
    X = []
    X.append(xday)
    X.append(xweekend)
    for r in range(0,9):
        line = fr.readline()
        re = line.strip('\n').split(',')
        data_str = map(float,re[1:])
        data_float = []
        for s in data_str:
            data_float.append(s)
        X.append((data_float))
    return np.array(X).T
#读取目标值
def readfile_oneshop_Y(fr):
    line = fr.readline()
    re = line.strip('\n').split(',')
    data_str = map(float,re[1:])
    data_float = []
    for s in data_str:
        data_float.append(s)
    return data_float
def Evaluation(pred,test):
    shop_err = 0
    for i in range(0,len(pred)):
        for p,t in zip(pred[i],test[i]):
            if (p+t) == 0.0:
                shop_err += 0.0
            else:
                shop_err += abs((p-t)/(p+t))
    total_err = shop_err/(len(pred)*len(pred[0]))
    return total_err
#shuchu
def output(fw,shopid,y_pre):
    y_pre_str = []
    y_pre_int = map(int,y_pre)
    y_pre_tostr = map(str,y_pre_int)
    for i in y_pre_tostr:
        y_pre_str.append(i)
    fw.write(str(shopid)+','+','.join(y_pre_str)+'\n')
##################    


while i<2000:
    # readfile
    Y = readfile_oneshop_Y(fr1)[-350:]
    
    y_train = Y
    x_test = xday[-14:]
    y_pre_diffmean = mean_normal_weekend_diff(Y,xday,xweekend,-21,-14)
    ###
    output(fw,i+1,y_pre_diffmean)
    print(i)
    i += 1

fr1.close()
#fr2.close()
fw.close()
