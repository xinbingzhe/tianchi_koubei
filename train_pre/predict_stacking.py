print('begin')
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesRegressor
from heamy.dataset import Dataset
#from sklearn.pipeline import Pipeline
from heamy.estimator import Regressor
from heamy.pipeline import ModelsPipeline
from sklearn.ensemble import AdaBoostRegressor
#from sklearn.feature_selection import RFECV
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
road1 = "e:/tianchi_koubei/mid_dataset/weather_pm_feature.txt"
road2 = "e:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "e:/tianchi_koubei/result/pre_stacking4_7_11.csv"
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
xweekday = create_weekday()
xweekend_list = []
xweekend = create_weekend(xweekend_list)
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
        if r < 3:
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


while i<2000:
    # readfile
    X = []
    Y = readfile_oneshop_Y(fr2)
    start = len(Y)
    #print(start)
    X = readfile_oneshop_X(fr1,xday,xweekend,xweekday,xholiday)[-(start+14):]
    x_train = X[:-14]
    y_train = Y[:]
    x_test = X[-14:-7]
    dataset = Dataset(x_train,y_train,x_test)
    ###
    params_lascv = {'n_jobs':4,'cv':8}
    lascv = LassoCV

    params_rcv = {'cv':8,'normalize':True,'gcv_mode':'auto','scoring':'neg_mean_squared_error'}
    rcv = RidgeCV
    
   
    params_rf = {'n_estimators':300,'max_depth':10,'min_samples_split':2,'warm_start':True,'n_jobs':4,'oob_score':True,'max_features':'log2'}
    rf = RandomForestRegressor
    
    params_ext = {'max_features':'log2','n_estimators':300,'max_depth':12,'oob_score': True, 'n_jobs':4,'bootstrap':True}
    ext = ExtraTreesRegressor
    ###
    
    params_gbrt = {'loss':'huber','n_estimators': 300,'max_depth':12,'learning_rate': 0.01, 'random_state': 3}
    gbrt = GradientBoostingRegressor
    ###
    params_las = {'alpha':1}
    ###stacking
    model_rf = Regressor(dataset=dataset, estimator=rf, parameters=params_rf,name='rf')
    #model_gbrt = Regressor(dataset=dataset, estimator=gbrt, parameters=params_gbrt,name='gbrt')
    model_ext = Regressor(dataset=dataset, estimator=ext, parameters=params_ext,name='ext')
    model_rcv = Regressor(dataset=dataset, estimator=rcv, parameters=params_rcv,name='rcv')
    #model_lascv = Regressor(dataset=dataset, estimator=lascv, parameters=params_lascv,name='lascv')
    pipeline = ModelsPipeline(model_rf,model_rcv,model_ext)
    stack_ds = pipeline.stack(k=5,seed=111)
    
    stacker = Regressor(dataset=stack_ds, estimator=Lasso, parameters=params_las)
    y_pre = stacker.predict()
    y_pre_last = np.append(y_pre,y_pre)
    y_pre_last[10]*1.08
    ###
    #loss_gbrt = Evaluation([y_pre_gbrt],[y_test])
    output(fw,i+1,y_pre_last)
    '''
    if loss_gbrt>0.015:
        output(fw_gbrt,i+1,y_pre_rf)
        fw_gbrt.write(str(i+1)+',gbrt,'+str(loss_gbrt)+'\n')
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
#print((Evaluation(stacking_pre,Y_test)))

#print(err_shop)