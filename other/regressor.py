import numpy as np 
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# 设置seed()每次生成的随机值都一样
np.random.seed(1337)

# 创建数据（散点图）
X = np.linspace(-1, 1, 200)  # 在-1和1之间返回均匀间隔的数据
np.random.shuffle(X)  # 打乱数据，改变X自身内容
Y = 0.5*X + 2 + np.random.normal(0, 0.05, (200,))  # 正太分布函数
# 绘图
plt.scatter(X, Y)
plt.show()

X_train, Y_train = X[:160], Y[:160]  # 训练集
X_test, Y_test = X[160:], Y[160:]  # 测试集

# 建立神经网络
model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

# 选择损失函数和优化器
model.compile(loss='mse', optimizer='sgd')

# 训练
print("Training----------")
for step in range(301):
    cost = model.train_on_batch(X_train, Y_train)
    if step % 100 == 0:
        print('train cost', cost)

# 测试
print('\nTesting----------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

# 绘制预测图
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_pred)
plt.plot(X_test, Y_pred)
plt.show()
