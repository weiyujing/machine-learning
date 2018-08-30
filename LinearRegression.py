import math
import numpy as np
import pandas as pd
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import datetime



data='D:/vs2017work/PythonApplication5/history_stock_data/DR华东电 - 副本.xls'
df=pd.read_excel(data,index_col='date')
outfile='D:/vs2017work/PythonApplication5/outabc.xls'

# 定义预测列变量，它存放研究对象的标签名
forecast_col = 'close'
# 定义预测天数，这里设置为所有数据量长度的1%
forecast_out = 10

#forecast_out = int(math.ceil(0.005*len(df)))
# 只用到df中下面的几个字段
df = df[['open', 'high', 'low', 'close', 'volume']]

# 构造两个新的列
# HL_PCT为股票最高价与最低价的变化百分比
df['HL_PCT'] = (df['high'] - df['low']) / df['low'] * 100.0
# HL_PCT为股票收盘价与开盘价的变化百分比
df['PCT_change'] = (df['close'] - df['open']) / df['open'] * 100.0
# 下面为真正用到的特征字段
df = df[['close', 'HL_PCT', 'PCT_change', 'volume']]
# 因为scikit-learn并不会处理空数据，需要把为空的数据都设置为一个比较难出现的值，这里取-9999，

df.fillna(-99999, inplace=True)
# 用label代表该字段，是预测结果
# 通过让与Adj. Close列的数据往前移动1%行来表示
df['label'] = df[forecast_col].shift(-forecast_out)

# 最后生成真正在模型中使用的数据X和y和预测时用到的数据数据X_lately
X = np.array(df.drop(['label'], 1))

# 我们还要做一些预处理。一般来说，您希望您的机器学习的特性在- 1到1之间。它可能没有什么大作用，但它通常加速处理和提高预测的准确率。
# 很幸运这个功能它被包含在scikit- learn的预处理模块中，您可以调用preprocessing.scale来缩放到你的X变量:
#X = preprocessing.scale(X)

# 上面生成label列时留下的最后1%行的数据，这些行并没有label数据，因此我们可以拿他们作为预测时用到的输入数据
X_lately = X[-forecast_out:] #这些行并没有label数据

X = X[:-forecast_out]    

# 抛弃label列中为空的那些行
df.dropna(inplace=True)
y = np.array(df['label'])


# 开始前，先X和y把数据分成两部分，一部分用来训练，一部分用来测试
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# 生成scikit-learn的线性回归对象
clf = LinearRegression(n_jobs=-1)
# 开始训练
clf.fit(X_train, y_train)
# 用测试数据评估准确性
accuracy = clf.score(X_test, y_test) 
print("Xtrain:",X_train,"Ytrain\n;",y_train)

print("\nXta:",X_test,"Yte\n;",y_test)
# 进行预测
forecast_set = clf.predict(X_lately)

print("\n",forecast_set, accuracy)
# 修改matplotlib样式
style.use('ggplot')

one_day = 86400
# 在df中新建Forecast列，用于存放预测结果的数据
df['Forecast'] = np.nan
# 取df最后一行的时间索引
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
next_unix = last_unix + one_day

# 遍历预测结果，用它往df追加行
# 这些行除了Forecast字段，其他都设为np.nan
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # [np.nan for _ in range(len(df.columns) - 1)]生成不包含Forecast字段的列表
    # 而[i]是只包含Forecast值的列表
    # 上述两个列表拼接在一起就组成了新行，按日期追加到df的下面
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df.to_excel(outfile)

# 开始绘图
df['close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


#outfile='D:/vs2017work/PythonApplication5/outabc.xls'

df=pd.read_excel(outfile,index_col='date')
data1=df['label'].shift(forecast_out)

#data1=df['label'][-forecast_out*2:-forecast_out].copy()
data3=data1[-forecast_out:].copy()
data2=df['Forecast'][-forecast_out:].copy()
data3.plot(kind='line')
data2.plot(kind='line')
plt.show()
