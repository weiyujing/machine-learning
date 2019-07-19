from sklearn.cluster import AffinityPropagation
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import random
from sklearn.datasets.samples_generator import make_blobs
import tushare as ts
'''
stock_list=[600535,600547,600332,600340,600362,600373,600383,600406,600435, 600436,600438]

白云山   (600332)	华夏幸福 (600340)	江西铜业 (600362)
中文传媒 (600373)	金地集团 (600383)	国电南瑞 (600406)
北方导航 (600435)	片仔癀   (600436)	通威股份 (600438)
蓝光发展 (600466)	中国动力 (600482)	中金黄金 (600489)
驰宏锌锗 (600497)	华丽家族 (600503)	方大炭素 (600516)
康美药业 (600518)	贵州茅台 (600519)	中天科技 (600522)
中铁工业 (600528)	天士力   (600535)	山东黄金 (600547)
厦门钨业 (600549)	恒生电子 (600570)	海螺水泥 (600585)
用友网络 (600588)	北大荒   (600598)	绿地控股 (600606)
东方明珠 (600637)	爱建集团 (600643)	城投控股 (600649)
福耀玻璃 (600660)	上海石化 (600688)	青岛海尔 (600690)
均胜电子 (600699)	三安光电 (600703)	物产中大 (600704)
中航资本 (600705)	文投控股 (600715)	中粮糖业 (600737)
辽宁成大 (600739)	华域汽车 (600741)	中航沈飞 (600760)
国电电力 (600795)	鹏博士   (600804)	山西汾酒 (600809)
600332,600340,600362,600373,600383,600406,600435, 600436,600438
600535,600547,600466,600482,600489,600497,600503,600516,600518,600519,600522,600528,600549,600570,600585,
600588,600598,600606,600637,600643,600649,600660,600688,600690,600699,600703,600704,
600705,600715,600737,600739,600741,600760,600795,600804,600809,

# # （1）数据读取
# df=ts.get_hist_data('600128')
#

ts.set_token('8791d37f92f5bf4babf2be26716ead550a4226c65b47f1718daa0a84')
pro = ts.pro_api()
for list in stock_list:
    df = pro.daily(ts_code=str(list)+'.SH', start_date='20160613', end_date='20190711')
    df.to_csv('E:/pycharm/workplace/7.11/stock_data/'+str(list)+'.csv',columns=['trade_date','open','high','low','close','vol','amount'])
'''

# 生成测试数据
#centers = [[1, 1], [-1, -1], [1, -1]]
# 生成实际中心为centers的测试样本300个，X是包含300个(x,y)点的二维数组，labels_true为其对应的真是类别标签
#X, labels_true = make_blobs(n_samples=100, centers=centers, cluster_std=0.5,
       #random_state=0)

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def draw_tp_line(ap_result):
    plt.figure(figsize=(14, 8))
    for i in range(len(ap_result)):
        plt.plot(ap_result[i], linewidth='2',color=randomcolor())

   #y = list(range(0, len(predicted)))
    plt.legend()
    plt.grid()  # 生成网格
    plt.show()
#获取数据
def gci(filepath):
    # 遍历filepath下所有文件，包括子目录

    filename = []
    file = []
    files = os.listdir(filepath)
    for fi in files:
        fi_d = os.path.join(filepath, fi)

        filename.append(os.path.basename(fi_d)[:-4])  # 返回文件名
        file.append(fi_d)
    return file, filename


def Get_AP(X):
    ap = AffinityPropagation(preference=-6).fit(X)
    cluster_centers_indices = ap.cluster_centers_indices_  # 预测出的中心点的索引，如[123,23,34]
    labels = ap.labels_  # 预测出的每个数据的类别标签,labels是一个NumPy数组
    print("aaa\n", labels, "\nbbb", cluster_centers_indices, "/ncc", X)

    n_clusters_ = len(cluster_centers_indices)  # 预测聚类中心的个数

    print('预测的聚类中心个数：%d' % n_clusters_)
    ap_result = [[] for i in range(n_clusters_)] #二维数组\
    k=0
    for i in range(len(labels)):
        ap_result[labels[k]].append(X[i])
        k=k+1
    for j in range(n_clusters_):
        draw_tp_line(ap_result[j])






    # 绘制图表展示
    import matplotlib.pyplot as plt
    from itertools import cycle



    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    # 循环为每个类标记不同的颜色
    for k, col in zip(range(n_clusters_), colors):
        # labels == k 使用k与labels数组中的每个值进行比较
        # 如labels = [1,0],k=0,则‘labels==k’的结果为[False, True]

        class_members = labels == k
        cluster_center = X[cluster_centers_indices[k]]  # 聚类中心的坐标
        print(class_members,"\n",cluster_center,)




        #ap_result=[]
        #ap_result.append(X[class_members])

  #      plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
  #      plt.plot(cluster_center[0], cluster_center[1], markerfacecolor=col,
   #              markeredgecolor='k', markersize=14)
   #     for x in X[class_members]:
   #         plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title('预测聚类中心个数：%d' % n_clusters_)




if __name__ == "__main__":
    filepath = 'E:/pycharm/workplace/7.11/stock_data'
    file, stockname = gci(filepath)
    X=[]

    for i in range(len(file)):
        stock = pd.read_csv(file[i], parse_dates=[0])['close']

        stock=list(stock)[-50:]
        normalised_window = [(float(p) / stock[0] - 1) for p in stock]
        X.append(normalised_window)

        stock=[]
    Get_AP(X)







