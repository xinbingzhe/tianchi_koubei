import numpy as np
road1 = "d:/tianchi_koubei/mid_dataset/count_user_view.txt"
way1 = 'r'
road2 = "d:/tianchi_koubei/mid_dataset/count_user_view_cor.txt"
way2 = 'w'

fr = open(road1,way1)
fw = open(road2,way2)

line = fr.readline()

i = 0
#shop = [d for d in range(1,2001)]
all = []
miss = [247,367,1752]
while i<3:
    re = line.strip('\n').split(',')
    shopid = int(re[0])
    view = map(int,re[1:])
    view_int = []
    for j in view:
        view_int.append(j)
    all.append(view_int)
    line = fr.readline()
    i += 1

all_t = np.array(all).T
all_mean = []
for e in all_t:
    all_mean.append(str(round(np.mean(e))))
for mi in miss:
    fw.write(str(mi)+','+','.join(all_mean)+'\n')
fr.close()
fw.close()
