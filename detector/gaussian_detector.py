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


    def train(self, train_data: pd.DataFrame, train_labels: pd.DataFrame) -> None:
        """
        Train the Gaussian Rumor Detector.
        Note that this method should be used after the original data has been PREPROCESSED

        Args:
            train_data (pd.DataFrame): The cleaned train data
            train_labels (pd.DataFrame): The labels that used to train the model
        """
        print("正在处理文本特征...")
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(train_data)

    def classify(self, text: str) -> int:
        # 谣言分类
        return True
