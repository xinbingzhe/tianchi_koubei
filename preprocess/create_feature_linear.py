print("begin")
import math
import numpy as np
from sklearn import preprocessing
road1 = "d:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
road2 = "d:/tianchi_koubei/mid_dataset/count_user_view.txt"
road3 = "d:/tianchi_koubei/mid_dataset/shop_OldNew_user_feature.txt"
way1 = 'r'
road4 = "d:/tianchi_koubei/mid_dataset/create_feature_linear.txt"
way2 = 'w'
'''
def create_xday(xday):
    for i in range(1,490):
        xday.append(i)
    return xday
xday_list = []
xday = create_xday(xday_list)
def create_weekend(xweekend):
    j = 3
    for i in range(1,490):
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
'''
min_max_scaler = preprocessing.MinMaxScaler()

fr1 = open(road1,way1)
fr2 = open(road2,way1)
fr3 = open(road3,way1)
fw = open(road4,way2)

line1 = fr1.readline()
line2 = fr2.readline()
line3 = fr3.readline()
i = 0

while i<2000:
    print('i',i)
    #前两个文件
    re1 = line1.strip('\n').split(',')
    re2 = line2.strip('\n').split(',')
    shopid = re1[0]
    pay_toint = map(float,re1[1:])
    view_toint = map(float,re2[1:])
    pay = []
    view = []
    for p,v in zip(pay_toint,view_toint):
        pay.append(p)
        view.append(v)
    pay_last7_mean = []
    pay_last7_sum = []
    pay_last14_last7_sumdiff = []
    pay_last14_last7_meandiff = []
    view_last14_sum = []
    for j in range(0,len(pay)):
        #print('j',j)
        if j>21:
            pay_last7_mean.append(np.mean(pay[j-21:j-14]))
            pay_last7_sum.append(sum(pay[j-21:j-14]))
        else:
            pay_last7_mean.append(np.mean(pay[0:j+1]))
            pay_last7_sum.append(sum(pay[0:j+1]))

        if j>28:
            pay_last14_last7_sumdiff.append(sum(pay[j-21:j-14])-sum(pay[j-28:j-21]))
            pay_last14_last7_meandiff.append(np.mean(pay[j-21:j-14])-np.mean(pay[j-28:j-21]))
            view_last14_sum.append(sum(view[j-28:j-14]))
        else:
            pay_last14_last7_sumdiff.append(sum(pay[round((j+1)/2):j+1])-sum(pay[0:round((j+1)/2)]))
            pay_last14_last7_meandiff.append(np.mean(pay[round((j+1)/2):j+1])-np.mean(pay[0:math.ceil((j+1)/2)]))
            view_last14_sum.append(sum(view[0:j+1]))
    try:
        pay_last7_mean_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(pay_last7_mean).reshape(-1, 1)).T[0])
        pay_last7_sum_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(pay_last7_sum).reshape(-1, 1)).T[0])
        pay_last14_last7_sumdiff_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(pay_last14_last7_sumdiff).reshape(-1, 1)).T[0])
        pay_last14_last7_meandiff_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(pay_last14_last7_meandiff).reshape(-1, 1)).T[0])
        view_last14_sum_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(view_last14_sum).reshape(-1, 1)).T[0])
    except (Exception,Warning):
        print("@@@")
        print(re1)
        print(re2)
        print(pay)
        print(view)
        print(np.array(pay_last7_mean))
        break
    pay_last7_mean_tostr = []
    pay_last7_sum_tostr = []
    pay_last14_last7_sumdiff_tostr = []
    pay_last14_last7_meandiff_tostr = []
    view_last14_sum_tostr = []
    for a,b,c,d,e in zip(pay_last7_mean_minmax_tostr,pay_last7_sum_minmax_tostr,pay_last14_last7_sumdiff_minmax_tostr,pay_last14_last7_meandiff_minmax_tostr,view_last14_sum_minmax_tostr):
        pay_last7_mean_tostr.append(a)
        pay_last7_sum_tostr.append(b)
        pay_last14_last7_sumdiff_tostr.append(c)
        pay_last14_last7_meandiff_tostr.append(d)
        view_last14_sum_tostr.append(e)

    fw.write(shopid+','+','.join(pay_last7_mean_tostr)+'\n')
    fw.write(shopid+','+','.join(pay_last7_sum_tostr)+'\n')
    fw.write(shopid+','+','.join(pay_last14_last7_sumdiff_tostr)+'\n')
    fw.write(shopid+','+','.join(pay_last14_last7_meandiff_tostr)+'\n')
    fw.write(shopid+','+','.join(view_last14_sum_tostr)+'\n')

    line1 = fr1.readline()
    line2 = fr2.readline()
    #用户信息数据
    print('user')
    re3 =  line3.strip('\n').split(',')
    new_str = map(float,re3[1:])
    line3 = fr3.readline()
    re4 =  line3.strip('\n').split(',')
    old_str = map(float,re4[1:])
    line3 = fr3.readline()
    re5 =  line3.strip('\n').split(',')
    newp_str = map(float,re5[1:])
    line3 = fr3.readline()
    re6 =  line3.strip('\n').split(',')
    oldp_str = map(float,re6[1:])
    new_int = []
    old_int = []
    newp_int = []
    oldp_int = []
    for n,o,npr,opr in zip(new_str,old_str,newp_str,oldp_str):
        new_int.append(n)
        old_int.append(o)
        newp_int.append(npr)
        oldp_int.append(opr)
    new_last7_sum = []
    old_last7_sum = []
    newp_last7_mean = []
    oldp_last7_mean = []
    for k in range(0,len(new_int)):
        if k>21:
            new_last7_sum.append(sum(new_int[k-21:k-14]))
            old_last7_sum.append(sum(old_int[k-21:k-14]))
            newp_last7_mean.append(np.mean(newp_int[k-21:k-14]))
            oldp_last7_mean.append(np.mean(oldp_int[k-21:k-14]))
        else:
            new_last7_sum.append(sum(new_int[0:k+1]))
            old_last7_sum.append(sum(old_int[0:k+1]))
            newp_last7_mean.append(np.mean(newp_int[0:k+1]))
            oldp_last7_mean.append(np.mean(oldp_int[0:k+1]))

    try:
        new_last7_sum_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(new_last7_sum).reshape(-1, 1)).T[0])
        old_last7_sum_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(old_last7_sum).reshape(-1, 1)).T[0])
        newp_last7_mean_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(newp_last7_mean).reshape(-1, 1)).T[0])
        oldp_last7_mean_minmax_tostr = map(str,min_max_scaler.fit_transform(np.array(oldp_last7_mean).reshape(-1, 1)).T[0])
    except (Exception,Warning):
        print("###")
        print(re3[1:])
        print(re4[1:])
        print(new_int)
        print(old_int)
        print(np.array(new_last7_sum))
        break
        
    new_last7_sum_str = []
    old_last7_sum_str = []
    newp_last7_mean_str = []
    oldp_last7_mean_str = []
    for ns,os,nps,ops in zip(new_last7_sum_minmax_tostr,old_last7_sum_minmax_tostr,newp_last7_mean_minmax_tostr,oldp_last7_mean_minmax_tostr):
        new_last7_sum_str.append(ns)
        old_last7_sum_str.append(os)
        newp_last7_mean_str.append(nps)
        oldp_last7_mean_str.append(ops)
    fw.write(shopid+','+','.join(new_last7_sum_str)+'\n')
    fw.write(shopid+','+','.join(old_last7_sum_str)+'\n')
    fw.write(shopid+','+','.join(newp_last7_mean_str)+'\n')
    fw.write(shopid+','+','.join(oldp_last7_mean_str)+'\n')

    line3 = fr3.readline()
    
    i += 1

fr1.close()
fr2.close()
fr3.close()
fw.close()
print("success")