#import pandas as pd
import numpy as np
print("begin")
road1 = "d:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'
road2 = "d:/tianchi_koubei/mid_dataset/count_user_view.txt"

fr1 = open(road1,way1)
fr2 = open(road2,way1)

line1 = fr1.readline()
line2 = fr2.readline()

user_7view = []
user_pay = []
corr = []
corr2 = []
i = 0

while line2:

    re1 = line1.strip('\n').split(',')
    re2 = line2.strip('\n').split(',')
    shop1 = re1[0]
    shop2 = re1[0]
    if shop1 == shop2:
        pay_char = map(int,re1[-200:]) 
        view_char = map(int,re2[-200:])
        pay_int = []
        view_int = []

        for p,v in zip(pay_char,view_char):
            pay_int.append(p)
            view_int.append(v)
        sdays_view = []
        for j in range(0,len(view_int)):
            if j>14:
                sdays_view.append(sum(view_int[j-14:j]))
            else:
                sdays_view.append(sum(view_int[0:j]))

        corr.append(np.corrcoef(pay_int,view_int)[0][1])
        corr2.append(np.corrcoef(pay_int,sdays_view)[0][1])
        line1 = fr1.readline()
        line2 = fr2.readline()
        i += 1
    else:
        line1 = fr1.readline()
        i += 1
print(len(corr))
print(np.mean(corr))
print(np.mean(corr2))