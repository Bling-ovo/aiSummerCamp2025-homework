# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# 加载数据，处理缺失值和日期格式
df = pd.read_csv(r'C:\Users\jl152\Desktop\aiSummerCamp2025-master\day3\assignment\data\household_power_consumption\household_power_consumption.txt', 
                 sep=';', 
                 na_values=['?'],
                 parse_dates={'datetime': ['Date', 'Time']},
                 dayfirst=True,
                 dtype={'Global_active_power': 'float',
                        'Global_reactive_power': 'float',
                        'Voltage': 'float',
                        'Global_intensity': 'float',
                        'Sub_metering_1': 'float',
                        'Sub_metering_2': 'float',
                        'Sub_metering_3': 'float'},
                 low_memory=False)

# 删除包含缺失值的行
df.dropna(inplace=True)

# 按时间划分训练集和测试集（以2009-12-31为界）
train = df[df['datetime'] <= '2009-12-31']
test = df[df['datetime'] > '2009-12-31']

# 数据归一化（仅使用训练集拟合）
scaler = MinMaxScaler(feature_range=(0, 1))
# 选择所有7个特征进行归一化
cols_to_scale = ['Global_active_power', 'Global_reactive_power', 'Voltage', 
                 'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

train_scaled = scaler.fit_transform(train[cols_to_scale])
test_scaled = scaler.transform(test[cols_to_scale])

# 创建时间序列数据
def create_sequences(data, look_back=60):
    """
    将时间序列数据转换为LSTM需要的三维格式
    参数:
        data: 归一化后的数据
        look_back: 使用过去多少个时间步预测未来
    返回:
        X: 输入序列 (样本数, 时间步, 特征数)
        y: 输出值 (下一个时间步的全局有功功率)
    """
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i])  # 取过去look_back个时间步
        y.append(data[i, 0])           # 预测下一个时间步的Global_active_power
    return np.array(X), np.array(y)

# 设置时间窗口为60分钟（1小时）
look_back = 60
X_train, y_train = create_sequences(train_scaled, look_back)
X_test, y_test = create_sequences(test_scaled, look_back)

# 构建LSTM模型
model = Sequential()
# 第一层LSTM（返回完整序列供下一层使用）
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))  # 防止过拟合
# 第二层LSTM（只返回最后一个时间步的输出）
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
# 输出层（预测单个值）
model.add(Dense(units=1))

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 训练模型（使用早停防止过拟合）
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,  # 20%训练数据用于验证
    callbacks=[early_stop],
    verbose=1
)

# 在测试集上评估
y_pred = model.predict(X_test)

# 将预测结果逆归一化（需要重建完整特征维度）
# 创建与原始特征维度相同的零矩阵
y_pred_expanded = np.zeros((len(y_pred), len(cols_to_scale)))
y_test_expanded = np.zeros((len(y_test), len(cols_to_scale)))

# 将预测值放在第一列（对应Global_active_power）
y_pred_expanded[:, 0] = y_pred.flatten()
y_test_expanded[:, 0] = y_test

# 逆归一化
y_pred_orig = scaler.inverse_transform(y_pred_expanded)[:, 0]
y_test_orig = scaler.inverse_transform(y_test_expanded)[:, 0]

# 计算均方根误差
rmse = np.sqrt(np.mean((y_pred_orig - y_test_orig)**2))
print(f'测试集RMSE: {rmse:.2f} 千瓦')

# 绘制前500个样本的预测结果对比
plt.figure(figsize=(15, 6))
plt.plot(y_test_orig[:500], label='实际值', alpha=0.7, linewidth=2)
plt.plot(y_pred_orig[:500], label='预测值', alpha=0.7, linestyle='--')
plt.title('家庭用电量预测 (LSTM模型)')
plt.xlabel('时间步 (每分钟)')
plt.ylabel('全局有功功率 (千瓦)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# 绘制训练损失曲线
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('模型训练损失曲线')
plt.xlabel('训练轮次')
plt.ylabel('均方误差')
plt.legend()
plt.grid(alpha=0.3)
plt.show()