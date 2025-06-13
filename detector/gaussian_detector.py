# This is a rumor detector that uses Gaussian distribution to detect.
from typing import Tuple
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.discriminant_analysis import (
    QuadraticDiscriminantAnalysis,
)
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.sparse import hstack


class GuassianRumorDetector:
    def __init__(self, reg_param=1e-6, random_state=19260817) -> None:
        """
        Initialize the Gaussian Rumor Detector.

        Args:
            reg_param (_type_, optional): The reg_param of the QDA model to avoid inverible matrix. Defaults to 1e-6.
        """
        self.random_state = random_state
        self.model = QuadraticDiscriminantAnalysis(reg_param=reg_param)
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.svd = TruncatedSVD(n_components=150, random_state=self.random_state)

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Preprocess the text by removing special characters and converting to lowercase.

        Args:
            text (str): The input text to preprocess.

        Returns:
            str: The preprocessed text.
        """
        text = text.lower()  # 转为小写
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # 移除URL
        text = re.sub(r'\@\w+|\#', '', text) # 移除@和#
        text = re.sub(r'[^a-z\s]', '', text) # 只保留英文字母和空格
        return text


    @staticmethod
    def preprocess_dataframe(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Preprocess the data for training.

        Args:
            data (pd.DataFrame): The raw data containing text and labels.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: The preprocessed text data and labels.
        """
        text_df = df['text'] = df['text'].fillna('')
        labels = df['label']
        text_df = text_df.apply(GuassianRumorDetector.preprocess_text)
        return text_df, labels


    def fit(self, train_data: pd.Series, train_labels: pd.Series) -> None:
        """
        Train the Gaussian Rumor Detector.
        Note that this method should be used after the original data has been PREPROCESSED

        Args:
            train_data (pd.DataFrame): The cleaned train data
            train_labels (pd.DataFrame): The labels that used to train the model
        """
        print("正在处理文本特征...")
        self.tfidf_vectorizer.fit(train_data)
        self.svd.fit(self.tfidf_vectorizer.transform(train_data))

        train_tfidf = self.tfidf_vectorizer.transform(train_data)
        train_svd = self.svd.transform(train_tfidf)
        print("正在训练模型...")
        self.model.fit(train_svd, train_labels)

    def classify(self, text: str) -> int:
        """
        Classify a single text.
        Args:
            text (str): The input text to classify.
        Returns:
            int: The predicted label for the input text.
        """
        processed_test = self.preprocess_text(text)
        test_tfidf = self.tfidf_vectorizer.transform([processed_test])
        test_svd = self.svd.transform(test_tfidf)
        prediction = self.model.predict(test_svd)
        return prediction[0]

    def classify_texts(self, texts: pd.Series) -> np.ndarray:
        """
        Classify a series of texts.

        Args:
            texts (pd.Series): The input texts to classify.

        Returns:
            np.ndarray: The predicted labels for the input texts.
        """
        processed_texts = texts.apply(self.preprocess_text)
        test_tfidf = self.tfidf_vectorizer.transform(processed_texts)
        test_svd = self.svd.transform(test_tfidf)
        predictions = self.model.predict(test_svd)
        return predictions
    
    
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