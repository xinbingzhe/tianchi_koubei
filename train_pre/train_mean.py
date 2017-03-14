print('begin')
import numpy as np
import matplotlib.pyplot as plt 
road1 = "E:/tianchi_koubei/mid_dataset/count_shop_pay_correct.txt"
way1 = 'r'
#road3 = "f:/tianchi_koubei/result/mean_result.csv"
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

#fw = open(road3,way2)

i = 0

#读取day
def readfile_oneshop_day(fr):
    line = fr.readline()
    re = line.strip('\n').split(',')
    data_str = map(float,re[1:])
    data_float = []
    for s in data_str:
        data_float.append(s)
    return data_float

def compute_change(day):
    cp = []
    for i in range(0,len(day)-1):
        cp.append(day[i]-day[i+1])
    return np.mean(cp)
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
def mean_normal_weekend_diff(pay_day,xday,xweekend,start,end):
    y_pre = []
    nor_pay = []
    weekend_pay = []
    for p,n,w in zip(pay_day[start:end],xday[start-14:end-14],xweekend[start-14:end-14]):
        if w == 1:
            weekend_pay.append(p)
        else:
            nor_pay.append(p)
    nor = np.mean (nor_pay)
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
#
def output(fw,shopid,y_pre):
    y_pre_str = []
    y_pre_int = map(int,y_pre)
    y_pre_tostr = map(str,y_pre_int)
    for i in y_pre_tostr:
        y_pre_str.append(i)
    fw.write(str(shopid)+','+','.join(y_pre_str)+'\n')
##################    
cp_total = []
y_tr = []
mean_total = []
diff_total = []

while i<2000:
    # readfile
    
    Y = readfile_oneshop_day(fr1)
    mean = np.mean(Y[-21:-14])
    cp = compute_change(Y[-21:-14])
    x_train = xday[-300:-14]
    y_train = Y[-300:-14]
    x_test = xday[-14:]
    y_test = Y[-14:]
    y_pre_cp = mean_cp_predict(x_test,mean,cp,0.5)
    y_pre_m = mean_predict(x_test,mean,1)
    y_pre_diff = mean_normal_weekend_diff(Y,xday,xweekend,-21,-14)

    cp_total.append(y_pre_cp)
    y_tr.append(y_test)
    mean_total.append(y_pre_m)
    diff_total.append(y_pre_diff)
    

    ###

    ###
    '''
    plt.scatter(x_train,y_train)
    plt.plot(x_test,y_pre_cp,color = 'red')
    path = "d://tianchi_koubei/fig/train_cp/"+str(i+1)+'.png'
    plt.savefig(path+".png")
    plt.clf()#清除图像，所有的都画到一起了'''
    
    #output(fw,i+1,y_pre_rf)
    print(i)
    i += 1
print(Evaluation(cp_total,y_tr))
print(Evaluation(mean_total,y_tr))
print(Evaluation(diff_total,y_tr))
fr1.close()
#fr2.close()
#fw.close()
