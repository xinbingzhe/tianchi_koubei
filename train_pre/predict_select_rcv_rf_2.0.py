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
road1 = "E:/tianchi_koubei/mid_dataset/weather_pm_feature.txt"
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "E:/tianchi_koubei/result/select_result2_withoutfeature_2.0.csv"
way2 = 'w'
def create_xday(xday):
    for i in range(1,504):
        xday.append(i)
    return xday
xday_list = []
xday = create_xday(xday_list)
def create_weekend(xweekend):
    j = 3
    qweekend = [68, 102, 190, 190, 229, 348, 446, 467]
    for i in range(1,504): 
        if j == 6 or j ==7:
            if i not in qweekend:
                xweekend.append(1)
                if j == 7:
                    j = 1
                else:
                    j += 1
            else:
                if j == 7:
                    j = 1
                else:
                    j += 1
                xweekend.append(0)
        else:
            xweekend.append(0)
            j += 1
    return xweekend
xweekend_list = []
xweekend = create_weekend(xweekend_list)
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
def create_holiday(xday):
    ho_f = []
    holiday = [51,65,66,67,88,89,93,94,95,96,97,98,99,134,177,178,185,186,187,222,223,224,225,226,227,228,229,237,277,278,279,306,307,308,345,346,347,406,443,444,445,459,460,461,462,463,464,465]
    for d in xday:
        if d not in holiday:
            ho_f.append(0)
        else:
            ho_f.append(1)
    return ho_f
xholiday = create_holiday(xday)
fr1 = open(road1,way1)
fr2 = open(road2,way1)
fw = open(road3,way2)
#fw_diffmean = open(road4,way2)

i = 0
#读取特征
def readfile_oneshop_X(fr,xday,xweekend,xweekday,xholiday):
    X = []
    X.append(xday)
    X.append(xweekend)
    X.append(xweekday)
    X.append(xholiday)
    '''
    for r in range(0,5):
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

un_pre = []

Y_test = []


#err_shop_rcv = []
while i<2000:
    # readfile
    X = []
    X = readfile_oneshop_X(fr1,xday,xweekend,xweekday,xholiday )[-364:]
    Y = readfile_oneshop_Y(fr2)[-350:]
    x_train = X[-289:-21]
    y_train = Y[-275:-7]
    x_val = X[-21:-14]
    y_val = Y[-7:]
    x_test = X[-14:]
    ###
    
    RigeLinearCV = linear_model.RidgeCV(cv=10)
    rcv = RigeLinearCV.fit(x_train,y_train)
    y_pre_rcv = rcv.predict(x_val)
    ###
    params_rf = {'n_estimators':500,'min_samples_split':2,'n_jobs':4}
    
    rf = RandomForestRegressor(**params_rf)
    rf.fit(x_train,y_train)
    y_pre_rf = rf.predict(x_val)
    ###
    y_pre_diff1 = mean_normal_weekend_diff(Y[-14:-7],xday[-28:-14],xweekend[-28:-14],-7,0)
    y_pre_diff2 = mean_normal_weekend_diff(Y[-21:-7],xday[-35:-14],xweekend[-35:-14],-14,0)
    ###
    
    
    #Y_test.append(y_test)
    #y_pre_diff = mean_normal_weekend_diff(Y,xday,xweekend,-21,-14)
    
    ###
    loss_rcv = Evaluation([y_pre_rcv],[y_val])
    loss_rf = Evaluation([y_pre_rf],[y_val])
    loss_diffmean1 = Evaluation([y_pre_diff1],[y_val])
    loss_diffmean2 = Evaluation([y_pre_diff2],[y_val])

    union = {loss_rcv:1,loss_rf:2,loss_diffmean1:3,loss_diffmean2:4}
    minloss = min(union.keys())
    minloss_num = union[minloss]

    x_un = np.concatenate((x_train,x_val),axis=0)
    y_un = np.append(y_train,y_val)

    #print(len(x_un))
    #print(len(y_un))
    if minloss_num == 2:
        rf.fit(x_un,y_un)
        y_pre = rf.predict(x_test)
        #plt.title('rf')
    elif minloss == 1:
        rcv.fi(x_un,y_un)
        y_pre =rcv.predict(x_test)
        #plt.title('rcv')
    elif minloss == 3:
        y_pre = mean_normal_weekend_diff(Y[-7:],xday[-21:],xweekend[-21:],-14,0)
    else:
        y_pre = mean_normal_weekend_diff(Y[-14:],xday[-28:],xweekend[-28:],-14,0)
        #plt.title('diff')
    
        #print(y_pre)
   
        #print(y_pre)
    output(fw,i+1,y_pre)
    

    '''
    plt.scatter(xday[-289:-14],y_un)
    #plt.scatter(xday[-14:],y_test,color = 'green')
    plt.scatter(xday[-14:],y_pre,color = 'green')
    plt.plot(xday[-14:],y_pre,color = 'red')
    plt.title(str())
    path = "E://tianchi_koubei/fig/select_pre/"+str(i+1)+'.png'
    plt.savefig(path+".png")
    plt.clf()#清除图像，所有的都画到一起了
    '''
    
    print(i)
    i += 1

fr1.close()
fr2.close()
fw.close()
#fw_gbrt.close()
#print((Evaluation(un_pre,Y_test)))
#print(Evaluation(diffmean_pre,Y_test))
#print(err_shop)