import numpy as np

from keras.datasets import boston_housing
from keras import models
from keras import layers

import matplotlib.pyplot as plt

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print(train_data[0])
print(train_data.shape)  # (404, 13)
print(test_data.shape)  # (102, 13)
print(train_targets[:10])
print(train_targets.shape)  # (404,)
print(test_targets.shape)  # (102,)


# 数据标准化，减去平均值再除以标准差
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


# 模型定义
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


#  K折交叉验证
def k_cross_validation():
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 100
    all_scores = []
    for i in range(k):
        print('processing fold #', i)
        # 准备验证数据，第k个分区的数据
        val_data = train_data[i*num_val_samples: (i+1)*num_val_samples]
        val_targets = train_targets[i*num_val_samples: (i+1)*num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [train_data[:i*num_val_samples], train_data[(i+1)*num_val_samples:]], axis=0
        )
        partial_train_targets = np.concatenate(
            [train_targets[:i*num_val_samples], train_targets[(i+1)*num_val_samples:]], axis=0
        )
        # 构建Keras模型(已编译)
        model = build_model()
        # 训练模式(静默模式，verbose=0)
        model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
        val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
        all_scores.append(val_mae)

    print(all_scores)
    print(np.mean(all_scores))


def k_cross_validation_new():
    k = 4
    num_val_samples = len(train_data) // k
    num_epochs = 500
    all_mae_histories = []
    for i in range(k):
        print('processing fold #', i)
        # 准备验证数据，第k个分区的数据
        val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
        val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

        # 准备训练数据，其他所有分区的数据
        partial_train_data = np.concatenate(
            [train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]], axis=0
        )
        partial_train_targets = np.concatenate(
            [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0
        )
        # 构建Keras模型(已编译)
        model = build_model()
        # 保存每折的验证结果
        history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                            epochs=num_epochs, batch_size=1, verbose=2)
        print(history.history.keys())
        mae_history = history.history['val_mae']
        all_mae_histories.append(mae_history)
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    # plt_plot(average_mae_history)
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt_plot(smooth_mae_history)


def plt_plot(average_mae_history: list):
    plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


# k_cross_validation()
# k_cross_validation_new()

model = build_model()
model.fit(train_data, train_targets, epochs=60, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print(test_mae_score)
