print('begin')
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.ensemble.forest import RandomForestRegressor
#from sklearn.ensemble.forest import RandomForestRegressor
#from sklearn.ensemble import GradientBoostingRegressor
road1 = "E:/tianchi_koubei/mid_dataset/feature_correct.txt"
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "E:/tianchi_koubei/result/err_shop_loss_select_split1.02_last14.txt"
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
def create_weekday():
    j = 3
    weekday = []
    for i in range(1,504):
        weekday.append(j)
        j += 1
        if j == 8:
            j = 1
    return weekday

xweekday = create_weekday()
xweekend_list = []
xweekend = create_weekend(xweekend_list)
#print(xweekday)
fr1 = open(road1,way1)
fr2 = open(road2,way1)
fw = open(road3,way2)
#fw_diffmean = open(road4,way2)

i = 0
#读取特征
def readfile_oneshop_X(fr,xday,xweekend,xweekday):
    X = []
    X.append(xday)
    X.append(xweekend)
    X.append(xweekday)
    '''
    for r in range(0,13):
        line = fr.readline()
        re = line.strip('\n').split(',')
        data_str = map(float,re[1:])
        data_float = []
        for s in data_str:
            data_float.append(s)
        X.append((data_float))
    '''
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
    for p,n,w in zip(pay_day,xday[:start],xweekend[:start]):
        if w == 1:
            weekend_pay.append(p)
        else:
            nor_pay.append(p)
    nor = np.mean(nor_pay)
    weekend = np.mean(weekend_pay)
    if end == 0:
        for d in xweekend[start:]:
            if d == 0:
                y_pre.append(nor)
            else:
                y_pre.append(weekend)
        return y_pre
    else:
        for d in xweekend[start:end]:
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
shop = []
un_pre = []
same_pre = []
Y_test = []
#err_shop_rcv = []
while i<100:
    # readfile
    X = []
    X = readfile_oneshop_X(fr1,xday,xweekend,xweekday)[-364:-14]
    Y = readfile_oneshop_Y(fr2)[-350:]
    x_train = X[-275:-7]
    y_train = Y[-275:-7]
    x_oob = X[-14:-7]
    y_oob = Y[-14:-7]
    x_test = X[-7:]
    y_test = Y[-7:]
    ###
    
    RigeLinearCV = linear_model.RidgeCV(cv=10)
    rcv = RigeLinearCV.fit(x_train,y_train)
    y_pre_rcv = rcv.predict(x_oob)
    ###
    params_rf = {'n_estimators':500,'max_depth':10,'min_samples_split':2,'n_jobs':4}
    
    rf = RandomForestRegressor(**params_rf)
    rf.fit(x_train,y_train)
    y_pre_rf = rf.predict(x_oob)
    ###
    y_pre_diff = mean_normal_weekend_diff(Y[-21:-14],xday[-35:-14],xweekend[-35:-14],-14,-7)
    ###
    
    
    Y_test.append(y_test)
    #y_pre_diff = mean_normal_weekend_diff(Y,xday,xweekend,-21,-14)
    
    ###
    loss_rcv = Evaluation([y_pre_rcv],[y_oob])
    loss_rf = Evaluation([y_pre_rf],[y_oob])
    loss_diffmean = Evaluation([y_pre_diff],[y_oob])

    union = {loss_rcv:1,loss_rf:2,loss_diffmean:3}
    minloss = min(union.keys())
    minloss_num = union[minloss]

    x_un = np.concatenate((x_train,x_oob),axis=0)
    y_un = np.append(y_train,y_oob)

    last3_me = np.median(Y[-35:-28])
    last2_me = np.median(Y[-28:-21])
    last1_me = np.median(Y[-21:-14])
    diff32_me = last3_me - 1.06*last2_me
    diff21_me = last2_me - 1.06*last1_me
    loss_split_rf = 100
    loss_split_rcv = 100
    loss_split_diffmean = 100
    if diff32_me >= 0 and diff21_me >= 0:
        loss_split_rf =  Evaluation([np.array(y_pre_rf)*0.92],[y_oob])
        loss_split_rcv = Evaluation([np.array(y_pre_rcv)*0.92],[y_oob])
        loss_split_diffmean = Evaluation([np.array(y_pre_diff)*0.92],[y_oob])


    #print(len(x_un))
    #print(len(y_un))
    if minloss_num == 2:
        #if diff32_me >= 0 and diff21_me >= 0:
        rf.fit(x_un,y_un)
        if loss_split_rf<loss_rf:
            y_pre = np.array(rf.predict(x_test))*0.92
        else:
             y_pre = rf.predict(x_test)
    elif minloss == 1:
        rcv.fi(x_un,y_un)
        if loss_split_rcv<loss_rcv:
            y_pre = np.array(rcv.predict(x_test))*0.92
        else:
            y_pre =rcv.predict(x_test)
    else:
        if loss_split_diffmean<loss_diffmean:
            y_pre_d = mean_normal_weekend_diff(Y[-21:-7],xday[-35:-14],xweekend[-35:-14],-7,0)
            y_pre = np.array(y_pre_d)*0.92
        else:
            y_pre = mean_normal_weekend_diff(Y[-21:-7],xday[-35:-14],xweekend[-35:-14],-7,0)

    same_pre.append(y_pre)
    '''
    
    last4 = np.mean(Y[-35:-28])
    last3 = np.mean(Y[-28:-21])
    last2 = np.mean(Y[-21:-14])
    last1 = np.mean(Y[-14:-7])
    last3_me = np.median(Y[-28:-21])
    last2_me = np.median(Y[-21:-14])
    last1_me = np.median(Y[-14:-7])
    diff43 = last4 - 1*last3
    diff32 = last3 - 1*last2
    diff21 = last2 - 1*last1
    diff32_me = last3_me - 1.1*last2_me
    diff21_me = last2_me - 1.1*last1_me
    per_m = (last3/last4 + last2/last3 + last1/last2)/3
    per_me = (last2_me/last3_me+last1_me/last2_me)/2
    per = (per_m+per_me)/2

    
    if diff32_me >= 0 and diff21_me >= 0 and diff32>= 0 and diff21>= 0:
        y_pre1 = np.array(y_pre[0:7])*0.92
        y_pre2 = np.array(y_pre[-7:])*0.92
        y_pre_t = np.concatenate((y_pre1,y_pre2),axis=0)
        output(fw, i+1, y_pre_t)
        fw.write(str(i+1)+','+str(Evaluation([y_pre1],[y_test]))+'\n')
        output(fw, i+1, y_pre)
        fw.write(str(i+1)+','+str(Evaluation([y_pre],[y_test]))+'\n')
        un_pre.append(y_pre1)
        shop.append([i+1,per])
    else:
        un_pre.append(y_pre)
    '''
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
#print((Evaluation(un_pre,Y_test)))
print((Evaluation(same_pre,Y_test)))
#print(shop)
#print(Evaluation(diffmean_pre,Y_test))
#print(err_shop)