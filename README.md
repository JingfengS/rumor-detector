# 谣言检测系统

本项目实现了两种用于文本数据谣言检测的机器学习模型。主要目标是将给定的文本分类为谣言或非谣言。我们通过两种不同的方法实现这一目标：逻辑回归分类器和二次判别分析（QDA）分类器。

## 项目结构

以下是本项目中关键文件的简要概述：

- `logistic_detector.py`: 使用**逻辑回归**实现的谣言检测器。
- `gaussian_detector.py`: 使用**二次判别分析 (QDA)** 实现的谣言检测器，该方法假设数据服从高斯分布。
- `data/`: 此目录应包含训练和验证数据集，分别命名为 `train.csv` 和 `val.csv`。每个 CSV 文件应包含一个名为 `text` 的列（包含文本数据）和一个名为 `label` 的列（包含相应的标签）。

## 调用说明

### 环境要求

本项目需要 Python 3 和以下库：

-   pandas
-   scikit-learn
-   numpy

使用 pip 安装这些依赖项：

```bash
pip install pandas scikit-learn numpy
```

### 数据准备

1.  在项目的根目录下创建一个名为 `data` 的文件夹。
2.  在 `data` 文件夹中，将训练数据放入名为 `train.csv` 的文件中，将验证数据放入 `val.csv` 中。
3.  确保两个 CSV 文件都有两列：`text` 和 `label`。`text` 列应包含要分类的文本，`label` 列应包含相应的二进制标签（例如，1 代表谣言，0 代表非谣言）。

### 运行检测器

从终端运行每个检测器脚本。

运行逻辑回归检测器：

```bash
python logistic_detector.py
```

运行高斯 (QDA) 检测器：

```bash
python gaussian_detector.py
```

执行后，每个脚本将：
1.  加载训练和验证数据。
2.  预处理文本数据。
3.  在训练集上训练相应的模型。
4.  在验证集上评估模型的准确性并打印结果。
5.  对一个示例文本进行分类并打印预测的标签。

## 作为库使用

在 Python 脚本中导入和使用 `LogisticRumorDetector` 和 `GuassianRumorDetector` 类。

### 示例: `LogisticRumorDetector`

```python
from detector.logistic_detector import LogisticRumorDetector
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_data_path = Path(__file__).parent.parent / 'data' / 'train.csv'
    val_data_path = Path(__file__).parent.parent / 'data' / 'val.csv'
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    detector = LogisticRumorDetector()
    train_texts, train_labels = detector.preprocess_dataframe(train_data)
    val_texts, val_labels = detector.preprocess_dataframe(val_data)
    detector.fit(train_texts, train_labels)
    
    predictions = detector.classify_texts(val_texts)
    accuracy = accuracy_score(val_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    sample_text = "Mr. Wang is the best teacher in SJTU."
    print("Classify a single text:", sample_text)
    prediction = detector.classify(sample_text)
    print(f"Predicted label for the sample text: {prediction}")
```

### 示例: `GuassianRumorDetector`

```python
from detector.gaussian_detector import GuassianRumorDetector
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_data_path = Path(__file__).parent.parent / 'data' / 'train.csv'
    val_data_path = Path(__file__).parent.parent / 'data' / 'val.csv'
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    detector = GuassianRumorDetector()
    train_texts, train_labels = detector.preprocess_dataframe(train_data)
    val_texts, val_labels = detector.preprocess_dataframe(val_data)
    detector.fit(train_texts, train_labels)
    
    predictions = detector.classify_texts(val_texts)
    accuracy = accuracy_score(val_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    sample_text = "Mr. Huang is the best teacher in SJTU."
    print("Classify a single text:", sample_text)
    prediction = detector.classify(sample_text)
    print(f"Predicted label for the sample text: {prediction}")
    

```

## 参数调整说明

通过修改检测器各自 `__init__` 方法中的参数来自定义其行为。

-   **`LogisticRumorDetector`**:
    -   `max_iter`: 逻辑回归求解器收敛的最大迭代次数。
    -   `random_state`: 用于随机数生成器的种子，以确保结果的可复现性。

-   **`GuassianRumorDetector`**:
    -   `reg_param`: QDA 模型的正则化参数，以提高数值稳定性。
    -   `random_state`: 用于随机数生成器的种子。
    -   还可以在 `__init__` 方法中更改 `TruncatedSVD` 的 `n_components`（主成分数量）。