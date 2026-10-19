# Spaceship Titanic 项目说明

## 项目简介

本项目用于完成 Kaggle 的 **Spaceship Titanic** 二分类挑战。任务背景是根据乘客的个人信息、旅行状态、舱位以及船上设施消费记录，预测乘客在飞船遭遇时空异常后是否被传送到了另一个维度。模型需要为测试集中的每一位乘客预测布尔型结果 `Transported`，Kaggle 使用分类准确率作为评分指标。

## 数据情况

- 训练集包含 8,693 名乘客和 14 个字段，其中 `Transported` 是预测目标。
- 测试集包含 4,277 名乘客和 13 个输入字段。
- 目标类别分布基本平衡，训练集中 `Transported=True` 约占 50.36%。
- 除乘客编号和目标列外，其余原始字段均存在少量缺失值。
- Kaggle 提交文件必须包含 `PassengerId` 和 `Transported` 两列。

## 技术方案

项目使用 Python、pandas、NumPy 和 scikit-learn 完成数据处理及建模，主要流程如下：

1. 从 `PassengerId` 中提取旅行组编号、组内编号、旅行组人数和是否独自出行。
2. 将 `Cabin` 拆分为甲板、舱号和舷侧，并构造舱位区段及舱位人数特征。
3. 从姓名、年龄和设施消费数据中构造家庭规模、年龄区间、总消费、对数消费及旅行组统计特征。
4. 结合业务规则和同组信息处理缺失值，例如处于休眠状态的乘客通常没有设施消费。
5. 使用按旅行组划分的五折 `StratifiedGroupKFold` 进行验证，避免同一旅行组同时出现在训练折和验证折中。
6. 分别训练逻辑回归、极端随机树和直方图梯度提升模型，再根据 OOF 预测融合三个模型。

## 当前结果

| 模型 | 五折 OOF 准确率 | OOF ROC-AUC |
|---|---:|---:|
| 逻辑回归 | 0.7947 | 0.8859 |
| 极端随机树 | 0.8073 | 0.8965 |
| 直方图梯度提升 | 0.8073 | 0.9010 |
| 融合模型 | **0.8119** | **0.9032** |

融合模型的权重为逻辑回归 0.2、极端随机树 0.3、直方图梯度提升 0.5，最终分类阈值为 0.490。本地验证结果表明，融合模型比最好的单模型准确率提高约 0.46 个百分点。当前结果为本地交叉验证成绩，Kaggle 公榜成绩需要上传提交文件后才能确认。

## 项目文件

```text
SOFTWARE II/
├─ raw data/                 # Kaggle 原始数据
├─ src/train.py              # 特征工程、训练、验证和预测主程序
├─ tests/test_project.py     # 数据与特征工程测试
├─ outputs/submission.csv    # 可直接上传 Kaggle 的提交文件
├─ outputs/metrics.json      # 完整模型指标和运行环境信息
├─ outputs/model_summary.md  # 模型结果摘要
├─ README.md                 # 环境配置和运行说明
├─ REPORT.md                 # 完整课程项目报告
└─ requirements.txt          # Python 依赖
```

## 运行方法

建议使用 Python 3.11 或 3.12。在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

如果只需要快速检查程序是否可以运行，可以执行：

```powershell
python -m src.train --quick --output-dir outputs-quick
```

## 当前完成状态

- 原始数据检查已完成。
- 缺失值处理和特征工程已完成。
- 三个模型的五折交叉验证及融合已完成。
- 单元测试和提交格式检查均已通过。
- 最终提交文件共 4,277 行，乘客编号顺序与官方样例完全一致，无重复编号或非法标签。
- 下一步可以将 `outputs/submission.csv` 上传 Kaggle，并记录公开榜与私有榜成绩。

