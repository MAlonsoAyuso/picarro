from __future__ import annotations

import sqlite3
from typing import (
    Any,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd

from picarro.util import check_item_types

ON_CONFLICT_OPTIONS = {
    "rollback",
    "abort",
    "fail",
    "ignore",
    "replace",
}


def insert_dataframe(
    conn: sqlite3.Connection,
    table_name: str,
    df: pd.DataFrame,
    on_conflict: str = "abort",
    ensure_columns=False,
):
    if df.index.name is None:
        if not isinstance(df.index, pd.RangeIndex):
            raise ValueError(
                f"Unnamed index should be RangeIndex, not {type(df.index)}"
            )
    else:
        df = df.reset_index()

    if ensure_columns:
        ensure_columns_exist(
            conn,
            table_name,
            {
                column: get_sqlite_data_type(dtype)
                for column, dtype in df.dtypes.items()
            },
        )

    return conn.executemany(
        build_insert_query(table_name, list(df.columns), on_conflict),
        df.itertuples(index=False),
    )


_SQLITE_TYPES_BY_NUMPY_KIND = {
    "i": "int",
    "f": "real",
    "M": "datetime",
    # "M": "int",
}


def get_sqlite_data_type(dtype: np.dtype) -> str:
    sqlite_type = _SQLITE_TYPES_BY_NUMPY_KIND.get(dtype.kind, None)
    if sqlite_type is None:
        raise ValueError(f"No sqlite data type defined for {dtype}")
    return sqlite_type


def ensure_columns_exist(
    conn: sqlite3.Connection, table_name: str, needed_cols: Mapping[str, str]
):
    existing_cols = dict(
        conn.execute(
            f"select name, type from pragma_table_info('{table_name}')"
        ).fetchall()
    )

    conflicting_cols = {
        column: dict(
            existing=existing_cols.get(column, None),
            needed=needed_cols.get(column, None),
        )
        for column in set(existing_cols) & set(needed_cols)
        if existing_cols.get(column, None) != needed_cols.get(column, None)
    }
    if conflicting_cols:
        raise ValueError(f"Conflicting columns: {conflicting_cols}")

    with conn:
        for column in set(needed_cols) - set(existing_cols):
            col_def = f"{column} {needed_cols[column]}"
            conn.execute(f"alter table {table_name} add column {col_def}")


def insert_mapping(
    conn: sqlite3.Connection,
    table_name: str,
    data: Mapping[str, Any],
    on_conflict: str = "abort",
):
    columns, values = zip(*data.items())
    return conn.execute(
        build_insert_query(table_name, columns, on_conflict),
        values,
    )


def build_insert_query(table_name: str, columns: Sequence, on_conflict: str) -> str:
    columns = check_item_types(columns, str)
    query = (
        f"insert or {on_conflict} into [{table_name}] ({list_bracketed(*columns)}) "
        f"values ({', '.join('?'*len(columns))})"
    )
    return query


Params = Union[Tuple[Any], Mapping[str, Any]]


def read_dataframe(
    conn: sqlite3.Connection,
    query: str,
    params: Optional[Params] = None,
) -> pd.DataFrame:
    cur = execute(conn, query, params)
    columns = get_colnames(cur)
    return pd.DataFrame(cur.fetchall(), columns=columns)


def execute(
    conn: sqlite3.Connection, query: str, params: Optional[Params] = None
) -> sqlite3.Cursor:
    return conn.execute(query) if params is None else conn.execute(query, params)


def get_colnames(cursor: sqlite3.Cursor) -> tuple[str]:
    # The column names of the last query.
    # cursor.description is an iterable of 7-tuples with the colname
    # as the first element (see sqlite3 docs)
    cols, *_ = zip(*cursor.description)
    return cols


def bracket(arg: str):
    return f"[{arg}]"


def list_items(*args: str) -> str:
    return ", ".join(args)


def list_bracketed(*args: str) -> str:
    return ", ".join(map(bracket, args))
