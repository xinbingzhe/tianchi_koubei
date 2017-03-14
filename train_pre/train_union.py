print('begin')
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFECV
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
road1 = "f:/tianchi_koubei/mid_dataset/create_feature_linear_pre.txt"
road2 = "f:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'
road3 = "f:/tianchi_koubei/result/adboost_result.csv"
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
    y_pre_tostr = map(str,y_pre)
    for i in y_pre_tostr:
        y_pre_str.append(i)
    fw.write(str(shopid)+','+','.join(y_pre_str)+'\n')
##################    

uni_pret = []
uni_yt = []


while i<2:
    # readfile
    X = []
    X = readfile_oneshop_X(fr1,xday,xweekend)[-364:]
    Y = readfile_oneshop_Y(fr2)[-350:]
    x_train = X[:-14]
    y_train = Y
    x_test = X[-14:]
    
    #
    #交叉验证的
    params = {'n_estimators': 800, 'loss': 'square', 
          'learning_rate': 0.01,  'random_state': 3}
    adb = AdaBoostRegressor(**params)
    rfecv = RFECV. RFECV(estimator=adb, step=1, cv=KFold(5),scoring = 'neg_mean_squared_error' )
    rfecv.fit((x_train,y_train))
    y_pre_rfecv = rfecv.predict(x_test)
    #
    #
    #
    #print(Evaluation([y_pre],[y_test]))
    
    RigeLinearCV = linear_model.RidgeCV(cv=3)
    clf1 = RigeLinearCV.fit(x_train,y_train)
    y_pre_cv = clf1.predict(x_test)
    #uni_pret.append(0.8*np.array(y_pre_adb)+0.2*np.array(y_pre_cv))
    #uni_pre.append(y_pre_adb)

    ###
    params_rf = {'n_estimators':800, 'min_samples_split': 1,'warm_start':True,'n_jobs':2}
    rf = RandomForestRegressor(**params_rf)
    rf.fit(x_train,y_train)
    y_pre_rf = rf.predict(x_test)
    ###
    uni_pre = 0.6*np.array(y_pre_adb)+0.1*np.array(y_pre_cv)+0.3*np.array(y_pre_rf)
    output(fw,i+1,uni_pre)
    print(i)
    i += 1

fr1.close()
fr2.close()
fw.close()
