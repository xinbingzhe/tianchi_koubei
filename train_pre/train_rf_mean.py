print('begin')
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
road1 = "E:/tianchi_koubei/mid_dataset/create_feature_withpre.txt"
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'
#road3 = "d:/tianchi_koubei/result/union_rf_mean_median_result.csv"
#way2 = 'w'
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
#fw = open(road3,way2)

i = 0
#读取特征
def readfile_oneshop_X(fr,xday,xweekend):
    X = []
    X.append(xday)
    X.append(xweekend)
    for r in range(0,13):
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

err_shop = []
Y_test = []
rf_pre = []
m2_pre = []
me2_pre = []
union_pre = []
while i<50:
    # readfile
    X = []
    X = readfile_oneshop_X(fr1,xday,xweekend)[-350:]
    Y = readfile_oneshop_Y(fr2)[-350:]
    x_train = X[:-14]
    y_train = Y[:-14]
    x_test = X[-14:]
    y_test = Y[-14:]


    ###
    params_rf = {'n_estimators':800, 'min_samples_split': 2,'warm_start':True,'n_jobs':4}
    rf = RandomForestRegressor(**params_rf)
    rf.fit(x_train,y_train)
    y_pre_rf = rf.predict(x_test)
    ###
    mean1 = np.mean(Y[-7:])
    mean2 = np.mean(Y[-14:])
    mean3 = np.mean(Y[-21:])
    mean4 = np.mean(Y[-28:])
    median1 = np.median(Y[-7:])
    median2 = np.median(Y[-14:])
    median3 = np.median(Y[-21:])
    median4 = np.median(Y[-28:])
    y_pre_m1 = mean_predict(x_test,mean1,1)
    y_pre_m2 = mean_predict(x_test,mean2,1)
    y_pre_m3 = mean_predict(x_test,mean3,1)
    y_pre_m4 = mean_predict(x_test,mean4,1)
    y_pre_me1 = mean_predict(x_test,median1,1)
    y_pre_me2 = mean_predict(x_test,median2,1)
    y_pre_me3 = mean_predict(x_test,median3,1)
    y_pre_me4 = mean_predict(x_test,median4,1)
    ###
    union = 0.6*np.array(y_pre_rf)+0.08*np.array(y_pre_m2)+0.04*np.array(y_pre_m1)+0.04*np.array(y_pre_m3)+0.04*np.array(y_pre_m4)+0.08*np.array(y_pre_me2)+0.04*np.array(y_pre_me1)+0.04*np.array(y_pre_me3)+0.04*np.array(y_pre_me4)
    Y_test.append(y_test)
    rf_pre.append(y_pre_rf)
    m2_pre.append(y_pre_m2)
    me2_pre.append(y_pre_me2)
    union_pre.append(union)
    #output(fw,i+1,union)
    print(i)
    i += 1

fr1.close()
fr2.close()
#fw.close()
print(Evaluation(rf_pre,Y_test))
print(Evaluation(m2_pre,Y_test))
print(Evaluation(me2_pre,Y_test))
print(Evaluation(union_pre,Y_test))
