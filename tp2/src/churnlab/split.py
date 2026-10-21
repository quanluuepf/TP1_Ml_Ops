"""Reproducible stratified train/validation/test split by customer."""

import pandas as pd
from sklearn.model_selection import train_test_split

from churnlab.data import ID_COLUMN, TARGET


def split_customers(df: pd.DataFrame, seed: int, valid_size: float = 0.2, test_size: float = 0.2):
    """Split rows into train/valid/test (default 60/20/20), stratified on Churn."""
    holdout = valid_size + test_size
    train_df, rest_df = train_test_split(
        df, test_size=holdout, stratify=df[TARGET], random_state=seed
    )
    valid_df, test_df = train_test_split(
        rest_df, test_size=test_size / holdout, stratify=rest_df[TARGET], random_state=seed
    )
    return train_df, valid_df, test_df


def allocation_table(train_df, valid_df, test_df) -> pd.DataFrame:
    """One row per customer with its split, sorted for a stable file hash."""
    parts = [
        pd.DataFrame({ID_COLUMN: part[ID_COLUMN].to_numpy(), "split": name})
        for name, part in [("train", train_df), ("valid", valid_df), ("test", test_df)]
    ]
    return pd.concat(parts).sort_values(ID_COLUMN).reset_index(drop=True)


def check_disjoint(train_df, valid_df, test_df) -> None:
    ids = [set(part[ID_COLUMN]) for part in (train_df, valid_df, test_df)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise ValueError("Customer IDs overlap across splits")
