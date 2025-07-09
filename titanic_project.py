# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 设置图表风格 - 修复样式名称问题
sns.set_style('whitegrid')  # 正确样式名称
sns.set_palette('pastel')

# %%
# 加载数据
file_path = r"C:\Users\jl152\Desktop\aiSummerCamp2025-master\day1\assignment\data\train.csv"
data = pd.read_csv(file_path)
df = data.copy()
print(f"成功加载数据，共{len(df)}条记录")

# %%
# 数据探索可视化
plt.figure(figsize=(15, 12))

# 1. 幸存者比例
plt.subplot(2, 2, 1)
survived_counts = df['Survived'].value_counts()
plt.pie(survived_counts, labels=['未幸存', '幸存'], autopct='%1.1f%%', 
        colors=['#ff9999','#66b3ff'], startangle=90)
plt.title('幸存者比例')
plt.axis('equal')

# 2. 性别分布
plt.subplot(2, 2, 2)
gender_counts = df['Sex'].value_counts()
sns.barplot(x=gender_counts.index, y=gender_counts.values, 
            palette=['#ff9999','#66b3ff'])
plt.title('乘客性别分布')
plt.ylabel('人数')

# 3. 年龄分布
plt.subplot(2, 2, 3)
sns.histplot(df['Age'].dropna(), bins=30, kde=True, color='#99c2a2')
plt.title('乘客年龄分布')
plt.xlabel('年龄')
plt.ylabel('人数')

# 4. 船舱等级分布
plt.subplot(2, 2, 4)
class_counts = df['Pclass'].value_counts().sort_index()
sns.barplot(x=class_counts.index, y=class_counts.values, palette='Blues_r')
plt.title('乘客船舱等级分布')
plt.xlabel('船舱等级')
plt.ylabel('人数')
plt.xticks([0,1,2], ['一等舱', '二等舱', '三等舱'])

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300)
plt.show()

# %%
# 删除不相关特征
df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True)
print("删除无关特征后剩余列:", df.columns.tolist())

# %%
# 处理缺失值
print("\n缺失值统计:")
print(df.isnull().sum())
print(f"\n处理前数据量: {len(df)}")
df.dropna(inplace=True)
print(f"删除缺失值后数据量: {len(df)}")

# %%
# 特征与生存率关系可视化
plt.figure(figsize=(15, 10))

# 1. 性别与生存率
plt.subplot(2, 2, 1)
sns.barplot(x='Sex', y='Survived', data=df, palette=['#ff9999','#66b3ff'])
plt.title('性别与生存率')
plt.ylabel('生存率')
plt.ylim(0, 1)

# 2. 船舱等级与生存率
plt.subplot(2, 2, 2)
sns.barplot(x='Pclass', y='Survived', data=df, palette='Blues_r')
plt.title('船舱等级与生存率')
plt.xlabel('船舱等级')
plt.ylabel('生存率')
plt.xticks([0,1,2], ['一等舱', '二等舱', '三等舱'])
plt.ylim(0, 1)

# 3. 年龄与生存率
plt.subplot(2, 2, 3)
sns.boxplot(x='Survived', y='Age', data=df, palette=['#ff9999','#66b3ff'])
plt.title('年龄与生存率')
plt.xlabel('是否幸存')
plt.ylabel('年龄')
plt.xticks([0,1], ['否', '是'])

# 4. 同行亲属数量与生存率
plt.subplot(2, 2, 4)
sns.barplot(x='SibSp', y='Survived', data=df, palette='viridis')
plt.title('同行兄弟姐妹/配偶数量与生存率')
plt.xlabel('数量')
plt.ylabel('生存率')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('survival_factors.png', dpi=300)
plt.show()

# %%
# 对分类特征进行独热编码
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# %%
# 分离特征和标签
X = df.drop('Survived', axis=1)
y = df['Survived']
print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# %%
# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20%数据作为测试集
    random_state=42   # 随机种子确保结果可复现
)
print(f"\n训练集大小: {len(X_train)} 条")
print(f"测试集大小: {len(X_test)} 条")

# %%
# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("\n特征标准化完成")

# %%
# 初始化三种分类模型
models = {
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

# 存储结果
results = {}
conf_matrices = {}

# 循环训练和评估模型
print("\n开始模型训练与评估...")
for name, model in models.items():
    print(f"\n训练 {name} 模型中...")
    # 训练模型
    model.fit(X_train_scaled, y_train)
    
    # 预测测试集
    y_pred = model.predict(X_test_scaled)
    
    # 评估性能
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # 存储结果
    results[name] = {
        'accuracy': accuracy,
        'report': report
    }
    conf_matrices[name] = confusion_matrix(y_test, y_pred)
    
    print(f"{name} 模型训练完成，准确率: {accuracy:.4f}")

# %%
# 模型性能可视化
# 1. 模型准确率比较
accuracies = [results[name]['accuracy'] for name in results]
model_names = list(results.keys())

plt.figure(figsize=(10, 6))
sns.barplot(x=model_names, y=accuracies, palette='viridis')
plt.title('模型准确率比较')
plt.ylabel('准确率')
plt.ylim(0.7, 0.9)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
plt.savefig('model_accuracy.png', dpi=300)
plt.show()

# 2. 混淆矩阵可视化
plt.figure(figsize=(15, 5))
for i, (name, matrix) in enumerate(conf_matrices.items()):
    plt.subplot(1, 3, i+1)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['未幸存', '幸存'], yticklabels=['未幸存', '幸存'])
    plt.title(f'{name} 混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=300)
plt.show()

# 3. 分类报告可视化（随机森林为例）
rf_report = results['Random Forest']['report']
report_data = []
lines = rf_report.split('\n')[2:-3]
for line in lines:
    row_data = line.split()
    if len(row_data) > 0:
        report_data.append(row_data)

report_df = pd.DataFrame(report_data[1:], columns=['类别', '精确率', '召回率', 'F1分数', '样本数'])
report_df = report_df.apply(pd.to_numeric, errors='ignore')

plt.figure(figsize=(10, 6))
sns.barplot(x='类别', y='F1分数', data=report_df, palette='coolwarm')
plt.title('随机森林模型分类性能 (F1分数)')
plt.ylim(0, 1)
for i, v in enumerate(report_df['F1分数']):
    plt.text(i, v + 0.03, f"{v:.2f}", ha='center')
plt.savefig('classification_report.png', dpi=300)
plt.show()

# %%
# 打印各模型性能
print("\n\n===== 模型评估结果 =====")
for name, res in results.items():
    print(f"\n----- {name} 模型性能 -----")
    print(f"准确率: {res['accuracy']:.4f}")
    print("分类报告:")
    print(res['report'])

print("\n所有模型评估完成！")
print("可视化图表已保存为: data_exploration.png, survival_factors.png, model_accuracy.png, confusion_matrices.png, classification_report.png")