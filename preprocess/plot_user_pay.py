print("begin")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import datetime

road1 = "d:/tianchi_koubei/mid_dataset/count_shop_pay.txt"
way1 = 'r'


datelist = pd.date_range('20150701', periods=290)

fr = open(road1,way1)

line = fr.readline()
paylist = []
i = 0
while line:
    re = line.strip('\n').split(',')
    pay_char = map(int,re[200:])
    pay_int = []
    for p in pay_char:
        pay_int.append(p)
    #plt.plot(datelist,pay_int)
    paylist.append(pay_int)
    line = fr.readline()
    #i += 1
paylist_np = np.array(paylist).T

#print(paylist_np)

df = pd.DataFrame(paylist_np,index=datelist)
df = df.cumsum()
plt.figure()
df.plot()

#plt.grid()

#plt.grid()
plt.show()

fr.close()
print("success")