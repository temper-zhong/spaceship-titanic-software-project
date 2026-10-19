# Spaceship Titanic 课程项目

本项目完成 Kaggle **Spaceship Titanic** 二分类任务：根据乘客资料预测 `Transported` 是否为 `True`。项目包括数据检查、特征工程、分组交叉验证、三个模型的比较与融合，以及可直接上传 Kaggle 的提交文件。

本次最终运行的 5 折分组 OOF 准确率为 **0.8119**，OOF ROC-AUC 为 **0.9032**。这是本地验证结果，不等同于 Kaggle 公榜分数。

## 项目结构

```text
SOFTWARE II/
├─ raw data/
│  ├─ train.csv
│  ├─ test.csv
│  └─ sample_submission.csv
├─ src/
│  └─ train.py
├─ tests/
│  └─ test_project.py
├─ outputs/
│  ├─ submission.csv
│  ├─ metrics.json
│  └─ model_summary.md
├─ REPORT.md
└─ requirements.txt
```

## 环境与运行

建议使用 Python 3.11 或 3.12。

首先从 Kaggle 的 [Spaceship Titanic 数据页面](https://www.kaggle.com/competitions/spaceship-titanic/data) 下载 `train.csv`、`test.csv` 和 `sample_submission.csv`，并将三个文件放入项目的 `raw data` 目录。Kaggle 原始数据和生成的提交文件不会提交到 GitHub。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.train
```

快速检查可使用：

```powershell
python -m src.train --quick --output-dir outputs-quick
```

正常运行会在 `outputs` 中生成：

- `submission.csv`：Kaggle 提交文件，列名和顺序与官方样例一致。
- `metrics.json`：环境版本、各折准确率、OOF 准确率、ROC-AUC、融合权重和阈值。
- `model_summary.md`：便于写入课程报告的模型比较表。

## 方法概要

1. 从 `PassengerId` 提取旅行组编号、组内编号和组大小。
2. 从 `Cabin` 提取甲板、舱号、舷侧及舱段。
3. 从姓名提取姓氏规模，并构造年龄、消费、同组均值等特征。
4. 使用休眠乘客消费为零等约束，并优先用同旅行组信息处理缺失值。
5. 使用 `StratifiedGroupKFold`，以旅行组为分组，避免同组成员同时进入训练折和验证折。
6. 比较逻辑回归、极端随机树和直方图梯度提升，再根据 OOF 结果选择融合权重与分类阈值。

完整设计、实验结果和局限性见 [REPORT.md](REPORT.md)。
