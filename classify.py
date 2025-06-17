import pickle
from pathlib import Path

# ----------------- 重要 -----------------
# 为了让 pickle 成功加载 .pkl 文件中的对象，
# 必须能找到该对象的原始类定义。
# 我们假设包含 GuassianRumorDetector 类的文件名为 gaussian_detector.py。
# 请确保该文件与此脚本在同一目录下。
try:
    from detector.gaussian_detector import GuassianRumorDetector
except ImportError:
    print("错误：无法导入 GuassianRumorDetector 类。")
    print(
        "请确保包含该类定义的 .py 文件（例如 'gaussian_detector.py'）与此脚本位于同一目录。"
    )
    exit()
# ----------------------------------------


class RumorClassifier:
    def __init__(self, model_path="detector/gaussian_detector.pkl"):
        """
        通过加载训练好的 .pkl 文件来初始化分类器。

        Args:
            model_path (str, optional): 指向 .pkl 模型的路径。
                                        默认为 'gaussian_rumor_detector.pkl'。
        """
        # 加载我们之前训练并保存的完整的 GuassianRumorDetector 对象
        try:
            with open(model_path, "rb") as f:
                self.detector = pickle.load(f)
            print("谣言检测模型已成功加载。")
        except FileNotFoundError:
            print(f"错误: 模型文件未找到，请检查路径 '{model_path}' 是否正确。")
            exit()

    def classify(self, text: str) -> int:
        """
        对输入的文本进行谣言检测。

        这个方法会调用加载好的 detector 对象内部的 classify 方法，
        该方法已经包含了文本预处理、特征转换和最终预测的全部流程。

        Args:
            text: 输入的文本字符串。

        Returns:
            int: 预测的类别（例如：0 代表非谣言，1 代表谣言）。
        """
        if not isinstance(text, str) or not text.strip():
            print("输入无效：请输入非空的文本字符串。")
            return -1  # 返回一个错误码或抛出异常

        # 直接使用加载的 detector 对象进行分类
        prediction = self.detector.classify(text)
        return int(prediction)


# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 创建分类器实例，模型会自动加载
    classifier = RumorClassifier(model_path="detector/gaussian_detector.pkl")

    # 2. 准备要检测的文本
    rumor_text = (
        "BREAKING: Scientists found that drinking bleach can cure the common cold."
    )
    non_rumor_text = (
        "The stock market showed moderate gains in today's trading session."
    )
    empty_text = ""

    # 3. 进行分类并打印结果
    prediction1 = classifier.classify(rumor_text)
    print(f"\n文本: '{rumor_text}'")
    print(f"预测结果: {prediction1} (1通常表示谣言)")

    prediction2 = classifier.classify(non_rumor_text)
    print(f"\n文本: '{non_rumor_text}'")
    print(f"预测结果: {prediction2} (0通常表示非谣言)")

    prediction3 = classifier.classify(empty_text)


