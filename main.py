from detector.gaussian_detector import GuassianRumorDetector
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    train_data_path = Path(__file__).parent / "data" / "train.csv"
    val_data_path = Path(__file__).parent / 'data' / 'val.csv'
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    detector = GuassianRumorDetector()
    train_texts, train_labels = detector.preprocess_dataframe(train_data)
    val_texts, val_labels = detector.preprocess_dataframe(val_data)
    detector.fit(train_texts, train_labels)
    
    predictions = detector.classify_texts(val_texts)
    accuracy = accuracy_score(val_labels, predictions)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
