from kedro.pipeline import node, Pipeline
from ai.pipelines.echo.node import echo, concat_dataframe, echo_with_time

def create_pipeline(**args) -> Pipeline:
    return Pipeline([
        node(
            name = "echo",
            func = echo,
            inputs = "message",
            outputs = "tl_report",
        ),
        node(
            name = "concat",
            func = concat_dataframe,
            inputs = ["message", "timestamp"],
            outputs = "master_echo",
        ),
        node(
            name = "echo with",
            func = echo_with_time,
            inputs = "master_echo",
            outputs = "master_report"
        )
    ])
