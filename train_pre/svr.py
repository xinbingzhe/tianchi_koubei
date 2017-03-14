print('begin')
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import linear_model
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import RFECV
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LassoLarsCV
from sklearn.linear_model import ElasticNet
from sklearn.metrics  import mean_squared_error
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR
from sklearn.linear_model import BayesianRidge
road1 = "E:/tianchi_koubei/mid_dataset/weather_pm_feature.txt"
road2 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "E:/tianchi_koubei/result/err_shop_loss_nusvr.txt"
#road4 ="E:/tianchi_koubei/result/err_shop_loss_gbrt.txt"
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
def create_weekday():
    j = 3
    weekday = []
    for i in range(1,504):
        weekday.append(j)
        j += 1
        if j == 8:
            j = 1
    return weekday
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
xweekday = create_weekday()
xweekend_list = []
xweekend = create_weekend(xweekend_list)

fr1 = open(road1,way1)
fr2 = open(road2,way1)
fw_rf = open(road3,way2)
#fw_gbrt = open(road4,way2)

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
        if  2<r < 4:
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
    for i in range(1,490):
        if data_float[i] != 0.0:
            start = i
            break
    return data_float[start-1:]
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
svr_pre = []
Y_test = []
#ext_pre7d = []
shop = []
err_shop_rf = []
err_shop_gbrt = []
while i<2000:
    # readfile
    X = []
    Y = readfile_oneshop_Y(fr2)
    start = len(Y)
    #print(start)
    X = readfile_oneshop_X(fr1,xday,xweekend,xweekday,xholiday)[-(start+14):-14]
    
    x_train = X[:-7]
    y_train = Y[:-7]
    x_test = X[-7:]
    y_test = Y[-7:]

    RigeLinearCV = linear_model.RidgeCV(cv=8,normalize=True,gcv_mode='auto',scoring='neg_mean_absolute_error')
    rcv = RigeLinearCV.fit(x_train,y_train)
    y_pre_rcv = rcv.predict(x_test)
    rcv_pre.append(y_pre_rcv)
    ###
    '''
    ###
    lacv = LassoLarsCV(normalize=True,cv=8)
    lacv.fit(x_train,y_train)
    y_pre_lacv = lacv.predict(x_test)
    lacv_pre.append(y_pre_lacv)
    ###
    
    params_rf = {'n_estimators':500,'max_depth':10,'min_samples_split':2,'warm_start':True,'n_jobs':4,'oob_score':True,'max_features':'log2'}
    rf = RandomForestRegressor(**params_rf)
    rf.fit(x_train,y_train)
    y_pre_rf = rf.predict(x_test)
    rf_pre.append(y_pre_rf)
    '''
    ###
    params_svr = {'max_features':'log2','n_estimators': 600,'max_depth':10,'oob_score': True, 'n_jobs':4,'bootstrap':True}
    svr =BayesianRidge(n_iter=300)
    svr.fit(x_train,y_train)
    y_pre_svr = svr.predict(x_test)
    svr_pre.append(y_pre_svr)
    #y_pre_ext1 = ext.predict(x_test[:-7])
    #y_pre_ext7d = np.append(y_pre_ext1,y_pre_ext1)
    #ext_pre7d.append(y_pre_ext7d)

    #print(y_pre_ext)
    #print(y_pre_ext1)
    ###
    '''
    params_gbrt = {'loss':'huber','n_estimators': 500,'max_depth':12,'learning_rate': 0.01, 'random_state': 3}
    gbrt = GradientBoostingRegressor(**params_gbrt)
    gbrt.fit(x_train,y_train)
    y_pre_gbrt = gbrt.predict(x_test)
    gbrt_pre.append(y_pre_gbrt)
    '''
    ###
    
    Y_test.append(y_test)

    ###
    '''
    loss_rf = Evaluation([y_pre_ext],[y_test])
    #loss_gbrt = Evaluation([y_pre_gbrt],[y_test])
    if loss_rf>0.09:
        output(fw_rf,i+1,y_pre_ext)
        shop.append(i+1)
        fw_rf.write(str(i+1)+',rf,'+str(loss_rf)+'\n')
        plt.scatter(xday[-(start+14):-21],y_train)
        plt.scatter(xday[-21:-14],y_test,color = 'green')
        plt.plot(xday[-21:-14],y_pre_ext,color = 'red')
        path = "e://tianchi_koubei/fig/rf_train_7d_0.9/"+str(i+1)+'.png'
        plt.savefig(path+".png")
        plt.clf()#清除图像，所有的都画到一起了'''
    

    '''
    if loss_gbrt>0.015:
        output(fw_gbrt,i+1,y_pre_rf)
        fw_gbrt.write(str(i+1)+',gbrt,'+str(loss_gbrt)+'\n')
        '''
    
    
    
    
    print(i)
    i += 1
#output(fw_rf,9999,shop)
fr1.close()
fr2.close()
fw_rf.close()
#fw_gbrt.close()
#print((Evaluation(rf_pre,Y_test)))
#print(Evaluation(gbrt_pre,Y_test))
print(Evaluation(rcv_pre,Y_test))
print(Evaluation(svr_pre,Y_test))
#print(Evaluation(ext_pre7d,Y_test))
#print(Evaluation(lacv_pre,Y_test))

#print(err_shop)