#最小二乘法
import numpy as np  ##科学计算库
import scipy as sp  ##在numpy基础上实现的部分算法库
import matplotlib.pyplot as plt ##绘图库
from scipy.optimize import leastsq ##引入最小二乘法算法
import pandas as pd

def draw_tp_line(y_test,predicted):
    plt.figure(figsize=(14, 8))
    plt.plot(predicted, label='Prediction', linewidth='3',color='r')
    plt.plot(y_test, label='true', color='b')
   #y = list(range(0, len(predicted)))
    plt.legend()
    plt.grid()  # 生成网格
    plt.show()

def func(p, x):
    '''
    summ = 0
    for i in range(60):#令k0，k1..k60，b=1
        cmd = "k%s =1" % i
        exec(cmd)
        eval("k%s" % i)
    b=1
    for j in range(60):  #ki*xi的和
        cmd = "summ=summ+k%s*X[j]" % j
        exec(cmd)
        eval("k%s" % j)

    return summ + b
    '''
    #print("x=",x)
    K = [p[i] for i in range(60)]
    b = p[60]
    #print("K=",K)
    new = K * x
    #print("K,B", K, b, new)
    nnn = []
    for j in range(61):
        sum = 0
        for i in range(60):
            sum = sum + new[j][i]
        nnn.append(sum + b)
        #print("nn",nnn)
    return nnn



##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
def error(p, x, y):

    return func(p, x) - y

def text_leastsq(k,b,x,y):##测试的，
    y_test=[]
    #print(x[0])
    y_test=k*x
    #print("y_test",y_test)
    nnn = []
    for j in range(60):
        sum = 0
        for i in range(60):
            sum = sum + y_test[j][i]
        nnn.append(sum + b)
    print("nnn",nnn)

    draw_tp_line(y[:60],nnn)
     #涨跌准确率
    kk=0
    for j in range(0, 19):
        if (float(y[j + 1]) - float(y[j])) * (float(nnn[j+1]) - float(y[j])) > 0:

            kk = kk + 1

    print("准确率",float(kk/20))
if __name__ == '__main__':
    for i in range(71):  # 定义自变量x0=[],x1=[]...
        cmd = "x%s = []" % i
        exec(cmd)
        eval("x%s" % i)
    stock = pd.read_csv('60_pre_out/600783.csv', parse_dates=[0])
    X = []#训练集
    X2=[]#测试集
    #xi为60维的第i条数据
    for j in range(1, 61):  # A1。。A60
        for ii in range(0, 71):  # 71条测试数据,61训练，10条测试
            cmd = "x%s.append(float(stock['A'+str(j)][ii]))" % ii
            exec(cmd)
            eval("x%s" % ii)
        '''
        x0.append(float(stock['A'+str(j)][0]))
        x1.append(float(stock['A' + str(j)][1]))
        x2.append(float(stock['A' + str(j)][2]))
    X.append(x0),X.append(x1),X.append(x2),X.append(x3),X.append(x4),X.append(x5),X.append(x6),
    '''
    #将xi循环加入X中
    for k in range(61):
        cmd = "X.append(x%s)" % k
        exec(cmd)
        eval("x%s" % k)
    for k in range(61,71):
            cmd = "X2.append(x%s)" % k
            exec(cmd)
            eval("x%s" % k)
    X=np.array(X)
    X2 = np.array(X2)
    Y=np.array([9.77,10.12,10.15 , 9.69 ,10.11 ,10.2 ,10.12, 10.13 , 9.85, 9.94, 10.04 , 9.86, 10.1, 9.96,10.16 ,10.17 ,10.24 ,9.74
                   ,9.71, 9.81 , 8.83, 8.61 , 8.55 , 8.49  , 8.52,  8.2, 8.6 , 9.29 ,  8.99 , 9.09, 9.09, 9.09 ,8.98 ,9.28 ,9.41,
                9.85 ,9.8 ,10.78 ,11.86,13.05 ,14.36 ,15.14 , 15.23 ,15.72 ,17.29,19.02 ,20.92 ,23.01 ,23.44 ,21.92 ,21.45
                ,19.31 ,19.47 ,19.36 ,19.24 ,18.42,17.79,18.15 ,18.57,16.71 ,15.88  ])
    Y2 = np.array([15.54,15.63,15.8,16.07,15.79,15.02,15.98,16.03,15.94,16.35])
    p0=[1for x in range(61) ]#给k0，k1，。。b赋值
    Para = leastsq(error, p0, args=(X, Y))

    # 读取结果
    K=Para[0][:60]
    b=Para[0][60]
    print("cost：" + str(Para[1]))

    text_leastsq(K, b, X, Y)
    #print("cost：" + str(Para[1]))
