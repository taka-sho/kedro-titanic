import pandas as pd

def echo(df: pd.DataFrame) -> str:
    """convert pandas' dataframe to text file"""

    tmp = df.name + " < " + df.message
    values = tmp.to_csv(index=False).replace("0\n", "")
    return values

def echo_with_time(df: pd.DataFrame) -> str:
    """convert pandas' dataframe to text file with timestamp"""

    tmp = df.name + " < " + df.message + "(" + df.Time + ")"
    values = tmp.to_csv(index=False).replace("0\n", "")
    return values

def concat_dataframe(df_msg: pd.DataFrame, df_time: pd.DataFrame) -> pd.DataFrame:
    """concat messages and timestamps and convert to text file"""

    concated = df_msg.join(df_time.set_index("name"), on="name")
    return concated