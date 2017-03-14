import datetime
def dateRange(start, end):
    days = (datetime.datetime.strptime(end, "%Y-%m-%d") - datetime.datetime.strptime(start, "%Y-%m-%d")).days + 1
    return [datetime.datetime.strftime(datetime.datetime.strptime(start, "%Y-%m-%d") + datetime.timedelta(i), "%Y-%m-%d") for i in range(days)]

dl = dateRange("2016-01-25","2016-03-01")
#print(dl)
dic = {}
for d in dl:
    dic[d]={'1':1}

for x in dic:
    print(dic[x]['1'])
#print(dic)
