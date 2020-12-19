import math
import pandas as pd
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
import boto3

def fetch_titanic_csv(params) -> pd.DataFrame:
    s3 = boto3.client("s3")
    df = s3.download_file(params["bucket_name"], params["file_path"], "data/01_raw/complete.csv")
    return df

def create_report_as_HTML(df: pd.DataFrame) -> str:
    report = ProfileReport(df)
    report_html = report.to_html()
    return report_html

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.Age = df.Age.fillna(df.Age.mean())
    df.Sex = df.Sex.replace({ "female": 0, "male": 1 })
    return df

def split_df(df, params) -> pd.DataFrame:
    testsize = params["testsize"]
    x_train, x_test, y_train, y_test = train_test_split(
        df[["Age", "Pclass", "Sex"]],
        df[["Survived"]],
        test_size = testsize,
        random_state = 0,
    )
    return [x_train, x_test, y_train, y_test]

def train_with_logistic(x_train, y_train):
    model = LogisticRegression(random_state = 0)
    model.fit(x_train, y_train)
    return model

def create_report_model_analysis(model, x_test, y_test) -> str:
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    fpr, tpr, thresholds = roc_curve(y_test, pred)
    auc_score = auc(fpr, tpr)

    return "acc: " + str(acc) + "\nf1: " + str(f1) + "\n" + "auc: " + str(auc_score)

    
    