print('begin')
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
#from sklearn.ensemble.forest import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
road1 = "E:/tianchi_koubei/mid_dataset/feature_correct.txt"
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "E:/tianchi_koubei/result/err_shop_loss_rcv_last14.txt"
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
#fw_diffmean = open(road4,way2)

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

def mean_normal_weekend_diff(pay_day,xday,xweekend,start,end):
    y_pre = []
    nor_pay = []
    weekend_pay = []
    for p,n,w in zip(pay_day[start:end],xday[start-14:end-14],xweekend[start-14:end-14]):
        if w == 1:
            weekend_pay.append(p)
        else:
            nor_pay.append(p)
    nor = np.mean(nor_pay)
    weekend = np.mean(weekend_pay)
    for d in xweekend[end-14:-14]:
        if d == 0:
            y_pre.append(nor)
        else:
            y_pre.append(weekend)
    return y_pre
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

rcv_pre = []

Y_test = []
err_shop_rcv = []
while i<100:
    # readfile
    X = []
    X = readfile_oneshop_X(fr1,xday,xweekend)[-364:-14]
    Y = readfile_oneshop_Y(fr2)[-350:]
    x_train = X[-175:-14]
    y_train = Y[-175:-14]
    x_test = X[-14:]
    y_test = Y[-14:]
    ###
    
    RigeLinearCV = linear_model.RidgeCV(cv=10)
    rcv = RigeLinearCV.fit(x_train,y_train)
    y_pre_rcv = rcv.predict(x_test)
    ###
    '''
    params_gbrt = {'loss':'huber','n_estimators': 800,'max_depth':12,'learning_rate': 0.01, 'random_state': 3}
    gbrt = GradientBoostingRegressor(**params_gbrt)
    gbrt.fit(x_train,y_train)
    y_pre_gbrt = gbrt.predict(x_test)
    gbrt_pre.append(y_pre_gbrt)'''
    ###
    rcv_pre.append(y_pre_rcv)
    Y_test.append(y_test)
    #y_pre_diff = mean_normal_weekend_diff(Y,xday,xweekend,-21,-14)
    
    ###
    loss_rcv = Evaluation([y_pre_rcv],[y_test])
    if loss_rcv>0.07:
        output(fw,i+1,y_pre_rcv)
        fw.write(str(i+1)+',gpr,'+str(loss_rcv)+'\n')
    
        
    '''
    plt.scatter(xday[-364:-14],y_train)
    plt.scatter(xday[-14:],y_test,color = 'green')
    plt.plot(xday[-14:],y_pre_rf,color = 'red')
    path = "d://tianchi_koubei/fig/rf_train/"+str(i+1)+'.png'
    plt.savefig(path+".png")
    plt.clf()#清除图像，所有的都画到一起了'''
    
    
    print(i)
    i += 1

fr1.close()
fr2.close()
fw.close()
#fw_gbrt.close()
print((Evaluation(rcv_pre,Y_test)))
#print(Evaluation(diffmean_pre,Y_test))
#print(err_shop)