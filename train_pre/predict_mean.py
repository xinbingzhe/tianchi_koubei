print('begin')
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
road1 = "f:/tianchi_koubei/mid_dataset/create_feature_linear_pre.txt"
road2 = "f:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'
road3 = "f:/tianchi_koubei/result/mean_result.csv"
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

fr1 = open(road1,way1)
fr2 = open(road2,way1)
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

def mean_cp_predict(x_test,mean,cp,n):
    y_pre_cp = []
    for i in range(0,len(x_test)):
        ypre = mean+n*cp
        mean = ypre
        y_pre_cp.append(ypre)
    return y_pre_cp
def mean_predict(x_test,mean,n):
    y_pre_m = []
    for i in range(0,len(x_test)):
        y_pre_m.append(n*mean)
    return y_pre_m


while i<2000:
    # readfile
    Y = readfile_oneshop_Y(fr2)[-350:]
    
    y_train = Y
    x_test = xday[-14:]
    
    mean = np.mean(Y[-10:])
    y_pre_m = mean_predict(x_test,mean,1)
    ###
    output(fw,i+1,y_pre_m)
    print(i)
    i += 1

fr1.close()
fr2.close()
fw.close()
