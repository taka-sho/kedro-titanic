from kedro.pipeline import Pipeline, node

from ai.pipelines.logistic_regression.node import create_report_as_HTML, preprocess, split_df, train_with_logistic, create_report_model_analysis, fetch_titanic_csv

def create_pipeline(**args) -> Pipeline:
    return Pipeline([
        node(
            name = "create HTML pre-report",
            func = create_report_as_HTML,
            inputs = "complete",
            outputs = "pre_data_validation",
        ),
        node(
            name = "create HTML post-report",
            func = create_report_as_HTML,
            inputs = "master_titanic",
            outputs = "post_data_validation",
        ),
        node(
            name = "preprocess dataframe",
            func = preprocess,
            inputs = "complete",
            outputs = "master_titanic",
        ),
        node(
            name = "split dataframe for train and validation",
            func = split_df,
            inputs = ["master_titanic", "parameters"],
            outputs = ["x_train", "x_test", "y_train", "y_test"]
        ),
        node(
            name = "train model using logistic regression",
            func = train_with_logistic,
            inputs = ["x_train", "y_train"],
            outputs = "logistic_model",
        ),
        node(
            name = "create model analysis report",
            func = create_report_model_analysis,
            inputs = ["logistic_model", "x_test", "y_test"],
            outputs = "model_analysis_report",
        )
    ])

def create_fetch_pipeline(**args) -> Pipeline:
    return Pipeline([
        node(
            name = "fetch titanic dataframe",
            func = fetch_titanic_csv,
            inputs  = ["parameters"],
            outputs = None,
        ),
    ])