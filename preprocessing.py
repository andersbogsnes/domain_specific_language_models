"""
Preprocessing steps for Input data
"""
from typing import Callable, List

import pandas as pd
import re
from nltk.tokenize import word_tokenize


def load_data(file_name: str = 'stackexchange_812k.csv.gz', nrows: int = None) -> pd.DataFrame:
    """
    Load data from CSV file on disk and convert to correct dtypes

    Parameters
    ----------
    file_name: str
        Path to data
    nrows: int
        Number of rows to load

    Returns
    -------
    pd.DataFrame
        DataFrame of loaded data
    """

    return pd.read_csv(file_name,
                       dtype={"post_id": "Int64",
                              "parent_id": "Int64",
                              "comment_id": "Int64",
                              "text": "string",
                              "category": "category"},
                       nrows=nrows)


def replace_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove instances of non-text words such as HTML tags or Latex equations

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame to clean

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with clean text
    """

    html_re = r'<.*?>'
    latex_re = r'\$.*?\$'
    newline_re = r'\n'
    user_re = r'@\w+'
    digits_re = r'[+-]?\d+'
    tags_re = r'\[.+\]'

    combined_re = re.compile('|'.join([html_re, latex_re, newline_re, user_re, digits_re, tags_re]))

    return df.assign(text=lambda x: x.text.str.replace(combined_re, ''))


def lowercase(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lowercases all text in the input DataFrame

    Parameters
    ----------
    df: pd.DataFrame
        Input DataFrame to modify

    Returns
    -------
    pd.DataFrame
        A copy of the input dataframe with all text lowercased
    """
    return df.assign(text=lambda x: x.text.str.lower())


def tokenize(df: pd.DataFrame,
             tokenizer: Callable[[str], List[str]] = word_tokenize) -> pd.DataFrame:
    """
    Creates a new column with tokens as

    Parameters
    ----------
    df: pd.DataFrame
        Input Dataframe to use for tokenizing
    tokenizer: Callable
        Function to use for tokenizing data. Defaults to nltk.tokenize.word_tokenize

    Returns
    -------
    pd.DataFrame
        A copy of the Input dataframe with a new column tokens, representing the split data
    """
    return df.assign(tokens=lambda x: x.text.apply(tokenizer))


def preprocess(file_name: str) -> pd.DataFrame:
    """Wrapper function for pipelining all datatransformations"""
    return (load_data(file_name)
            .pipe(lowercase)
            .pipe(replace_text)
            .pipe(tokenize)
            )


if __name__ == '__main__':
    cleaned = preprocess('stackexchange_812k.csv.gz')
    cleaned.to_csv('cleaned.csv', index=False)
