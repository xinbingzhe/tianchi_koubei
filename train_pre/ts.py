print('begin')
import __future__
import numpy as np
import matplotlib.pyplot as plt 
import statsmodels.api as sm
import pandas as pd
road = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
road3 = "E:/tianchi_koubei/result/err_shop_loss_ts10.txt"
way2 = 'w'

fr = open(road,way1)
fw = open(road3,way2)

def readfile_oneshop_Y(fr):
    line = fr.readline()
    re = line.strip('\n').split(',')
    data_str = map(float,re[1:])
    data_float = []
    for s in data_str:
        data_float.append(s)
    return data_float
def output(fw,shopid,y_pre):
    y_pre_str = []
    y_pre_int = map(int,y_pre)
    y_pre_tostr = map(str,y_pre_int)
    for i in y_pre_tostr:
        y_pre_str.append(i)
    fw.write(str(shopid)+','+','.join(y_pre_str)+'\n')
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
i = 0
ts_pre = []
Y_test = []

while i<1000:

    Y = readfile_oneshop_Y(fr)
     
    y_train = Y[-150:-14]
    y_test = Y[-14:]
    days = len(y_train)
    dta=pd.Series(y_train)
    dta.index = pd.Index(pd.date_range('20160604', periods=days))

    arma = sm.tsa.ARMA(dta,(1,0)).fit()
    arma_pre = arma.predict('20161018', '20161031', dynamic=True)
    y_pre = arma_pre.values
    ts_pre.append(y_pre)
    Y_test.append(y_test)
    loss = Evaluation([y_pre],[y_test])
    output(fw,i+1,y_pre)
    fw.write(str(i+1)+',ts,'+str(loss)+'\n')

    i += 1
fr.close()
fw.close()
print((Evaluation(ts_pre,Y_test)))