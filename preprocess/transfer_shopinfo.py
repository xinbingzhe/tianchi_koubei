print("begin")
road1 = "d:/tianchi_koubei/origin_dataset/dataset/dataset/shop_info_gbk.txt"
way1 = 'rb'
road2 = "d:/tianchi_koubei/mid_dataset/transfer_shopinfo.txt"
way2 = 'w'

fr = open(road1,way1)
fw = open(road2,way2)
line = fr.readline()
scorel = []
commentl = []

shopid_list = []
cate1_list = []
cate2_list = []
cate2_list = []
i = 0

while i<10:

    line_tostr = line.decode('gbk','ignore')
    re = line_tostr.strip('\n').split(',')

    shopid = re[0]
    city_name = re[1]
    cate1 = re[-3]
    cate2 = re[-2]
    cate3 = re[-1]
    score = re[4]
    comment = re[5]

    if 
    print(line_tostr)
    #print(re)

    line = fr.readline()
    i += 1

fr.close()
fw.close()