# 谣言检测系统

本项目实现了两种用于文本数据谣言检测的机器学习模型。主要目标是将给定的文本分类为谣言或非谣言。我们通过两种不同的方法实现这一目标：逻辑回归分类器和二次判别分析（QDA）分类器。

## 项目结构

以下是本项目中关键文件的简要概述：

- `logistic_detector.py`: 使用**逻辑回归**实现的谣言检测器。
- `gaussian_detector.py`: 使用**二次判别分析 (QDA)** 实现的谣言检测器，该方法假设数据服从高斯分布。
- `data/`: 此目录应包含训练和验证数据集，分别命名为 `train.csv` 和 `val.csv`。每个 CSV 文件应包含一个名为 `text` 的列（包含文本数据）和一个名为 `label` 的列（包含相应的标签）。

## 方法论

两种检测器都遵循类似的文本分类流程：

1.  **文本预处理**: 对输入文本进行清洗，步骤如下：
    * 将所有字符转换为小写。
    * 移除 URL、@提及 (`@`) 和话题标签 (`#`)。
    * 过滤掉所有非英文字母的字符。
    * 这确保了输入模型的文本是标准化的，并且不含可能影响分类准确性的噪音。

2.  **特征提取**: 将预处理后的文本转换为机器学习模型可以理解的数值格式。这是通过**词频-逆文档频率 (TF-IDF)** 向量化技术实现的。TF-IDF 将每个文本表示为一个数值向量，其中每个值对应一个词在文本中相对于整个语料库的重要性。

3.  **模型训练与分类**: 向量化的文本数据随后被用于训练相应的分类模型。

### `logistic_detector.py`

该检测器采用**逻辑回归**模型。逻辑回归是一种广泛用于二元分类任务的线性模型。它对给定输入属于特定类别的概率进行建模。在此项目中，它根据 TF-IDF 特征学习区分谣言和非谣言文本。

### `gaussian_detector.py`

该检测器采用了一种更复杂的方法：

1.  **降维**: 在 TF-IDF 向量化之后，特征空间的维度可能非常高。为了解决这个问题，我们使用了**截断奇异值分解 (TruncatedSVD)**。TruncatedSVD 是一种矩阵分解技术，可以在保留最重要信息的同时减少特征数量。在此实现中，特征空间被降至 150 个维度。在文本处理的背景下，这也被称为潜在语义分析 (LSA)。

2.  **分类**: 降维后的特征集被输入到**二次判别分析 (QDA)** 分类器中。QDA 是一个生成模型，它假设每个类别的数据都服从高斯分布。与线性判别分析 (LDA) 不同，QDA 不假设每个类别的协方差矩阵是相同的，这使得它能够形成更灵活的二次决策边界。

##快速入门

### 环境要求

本项目需要 Python 3 和以下库：

-   pandas
-   scikit-learn
-   numpy

您可以使用 pip 安装这些依赖项：

```bash
pip install pandas scikit-learn numpy
```

### 数据准备

1.  在项目的根目录下创建一个名为 `data` 的文件夹。
2.  在 `data` 文件夹中，将您的训练数据放入名为 `train.csv` 的文件中，将验证数据放入 `val.csv` 中。
3.  确保两个 CSV 文件都有两列：`text` 和 `label`。`text` 列应包含您要分类的文本，`label` 列应包含相应的二进制标签（例如，1 代表谣言，0 代表非谣言）。

### 运行检测器

您可以直接从终端运行每个检测器脚本。

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

您也可以在自己的 Python 脚本中导入和使用 `LogisticRumorDetector` 和 `GuassianRumorDetector` 类。

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
    
    sample_text = "Trump is the worst president in the world."
    print("Classify a single text:", sample_text)
    prediction = detector.classify(sample_text)
    print(f"Predicted label for the sample text: {prediction}")
    

```

## 自定义

您可以通过修改检测器各自 `__init__` 方法中的参数来自定义其行为。

-   **`LogisticRumorDetector`**:
    -   `max_iter`: 逻辑回归求解器收敛的最大迭代次数。
    -   `random_state`: 用于随机数生成器的种子，以确保结果的可复现性。

-   **`GuassianRumorDetector`**:
    -   `reg_param`: QDA 模型的正则化参数，以提高数值稳定性。
    -   `random_state`: 用于随机数生成器的种子。
    -   您还可以在 `__init__` 方法中更改 `TruncatedSVD` 的 `n_components`（主成分数量）。

欢迎您尝试调整这些参数，甚至更换底层的机器学习模型，以观察其在您的数据集上的性能变化。