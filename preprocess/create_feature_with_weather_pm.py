print('begin')
import numpy as np
road1 = "E:/tianchi_koubei/mid_dataset/shop_weather_decoder.txt"
road2 = "E:/tianchi_koubei/mid_dataset/pm_feature.csv"
way1 = 'r'
road3 = "E:/tianchi_koubei/mid_dataset/weather_pm_feature.txt"
way2 = 'w'

fr1 = open(road1,way1)
fr2 = open(road2,way1)
fw = open(road3,way2)

def read_write_weather(fr,fw,i):
    feature = []
    for r in range(0,519):
        line = fr.readline()
        if r < 503:
            re = line.strip('\n').split(',')
            data = [re[3],re[4],re[5],re[6]]
            feature.append(data)
    #print(len(feature))
    output = np.array(feature).T
    #print(output)
    #print(len(output))
    for o in output:
        fw.write(str(i)+','+','.join(o)+'\n')
def read_write_pm(fr,fw):
    line = fr.readline()
    re = line.strip('\n').split(',')[:504]
    #print(len(re))
    fw.write(','.join(re)+'\n')

i = 0
while i<2000:
    read_write_weather(fr1,fw,i+1)
    read_write_pm(fr2,fw)

    i+=1
fr1.close()
fr2.close()
fw.close()
    