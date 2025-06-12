"""Make test tables. Couple of approaches. GPT."""

from datetime import datetime, timedelta
import random
from random import randint, uniform, sample

import pandas as pd
from faker import Faker

# Simulate a list of words for column name generation
words = [
    "transaction", "identifier", "processing", "timestamp", "user", "account", "description",
    "amount", "balance", "location", "currency", "status", "failure", "note", "reference",
    "operation", "duration", "estimate", "category", "filename", "extension", "type", "project",
    "client", "supplier", "remark", "address", "email", "comment", "entry", "premium",
    "loss ratio", 'expense ratio', "combined ratio", 'loss date'
]

fake = Faker()


def make_column_name():
    # choices with replacement -> sample
    return " ".join(random.sample(words, k=random.randint(1, 5)))


def make_text_blob():
    return " ".join(sample(words, randint(10, 25)))


def make_test_dataframe(n_rows, n_cols):
    col_types = random.choices(["int", "float", "str", "date"], k=n_cols)
    data = {}
    for _ in range(n_cols):
        dtype = col_types.pop(0)
        col_name = make_column_name() + f' ({dtype})'
        if dtype == "int":
            data[col_name] = [random.randint(0, 10000) if random.random() > 0.1 else None for _ in range(n_rows)]
        elif dtype == "float":
            data[col_name] = [round(random.uniform(0, 1e4), 3) if random.random() > 0.1 else None for _ in range(n_rows)]
        elif dtype == "str":
            data[col_name] = [fake.sentence(nb_words=random.randint(2, 8)) if random.random() > 0.1 else None for _ in range(n_rows)]
        elif dtype == "date":
            start = datetime(2015, 1, 1)
            data[col_name] = [
                (start + timedelta(days=random.randint(0, 4000))).date().isoformat()
                if random.random() > 0.1 else None for _ in range(n_rows)
            ]
    return pd.DataFrame(data)


def make_dataframe_set(n):
    """Sample dataframes with n rows."""
    def rand_date():
        start = datetime(2000, 1, 1)
        return [(start + timedelta(days=randint(0, 10000))).strftime("%Y-%m-%d") for _ in range(n)]

    def rand_float():
        return [f"{uniform(0, 10000):.3f}" for _ in range(n)]

    def rand_int():
        return [str(randint(0, 5000)) for _ in range(n)]

    def rand_text():
        return [make_text_blob() for _ in range(n)]

    def rand_filename():
        return [f"{'_'.join(sample(words, randint(2, 5)))}.pdf" for _ in range(n)]

    def col(colfunc, allow_missing=False):
        vals = colfunc()
        if allow_missing:
            for i in range(randint(1, 3)):
                vals[randint(0, len(vals) - 1)] = ''
        return vals

    dfs = {}
    dfs["floats dates filenames"] = pd.DataFrame({
        make_column_name(): col(rand_float),
        make_column_name(): col(rand_date),
        make_column_name(): col(rand_filename),
        make_column_name(): col(rand_int),
        make_column_name(): col(rand_float, allow_missing=True),
    })

    dfs["dense text and numbers"] = pd.DataFrame({
        make_column_name(): col(rand_text),
        make_column_name(): col(rand_float),
        make_column_name(): col(rand_int),
        make_column_name(): col(rand_text),
        make_column_name(): col(rand_date),
        make_column_name(): col(rand_float, allow_missing=True),
    })

    dfs["mixed data with missing"] = pd.DataFrame({
        make_column_name(): col(rand_float, allow_missing=True),
        make_column_name(): col(rand_text, allow_missing=True),
        make_column_name(): col(rand_int, allow_missing=True),
        make_column_name(): col(rand_date, allow_missing=True),
        make_column_name(): col(rand_filename, allow_missing=True),
    })

    dfs["long header names"] = pd.DataFrame({
        "Detailed Instrumentation Configuration Summary": col(rand_text),
        "Archive Metadata Extraction Date Field": col(rand_date),
        "Overview Record Approximation Notes": col(rand_text),
        "Velocity Gradient Approximation Float": col(rand_float),
        "Pressure Summary Int Field": col(rand_int),
    })

    dfs["file-centric record"] = pd.DataFrame({
        make_column_name(): col(rand_filename),
        make_column_name(): col(rand_date),
        make_column_name(): col(rand_text),
        make_column_name(): col(rand_float),
        make_column_name(): col(rand_int),
        make_column_name(): col(rand_date),
        make_column_name(): col(rand_filename, allow_missing=True),
    })

    return dfs


def make_manual_tests():
    """Five handwritten test tables."""
    df1 = pd.DataFrame({
        "Consideration of Consequences": ["A rather long text value that could wrap badly.", "Short", "A second problematic entry with spaces."],
        "Probability": ["Likely", "Unlikely", "Moderate"],
        "Expected Value": ["High", "Low", "Moderate"]
    })

    df2 = pd.DataFrame({
        "event_date": ["2024-12-28", "2025-01-05", "2031-06-21"],
        "timestamp": ["2024-12-28T14:23:00", "2025-01-05T09:12:45", "2031-06-21T23:59:59"],
        "transaction_code": ["ABC-1001-ZZ", "XYZ-2048-AA", "LONG-CODE-2025-EXTREME"]
    })

    df3 = pd.DataFrame({
        "notes": [
            "Item 1: delivered; ready for invoice.",
            "Warning -- unit may be faulty?",
            "Check: power supply (see page 42)"
        ],
        "status": ["✓", "✗", "↺"],
        "path": [
            "/usr/local/bin/run.sh",
            "C:\\Program Files\\App\\main.exe",
            "~/Documents/projects/final-report.pdf"
        ]
    })

    df4 = pd.DataFrame({
        "Serial": ["A123B456", "X987Y654", "Z000Z111"],
        "MD5 Hash": [
            "a5c3e1d7f2b9c3d6f1e4a9b3c7d1e2f3",
            "9f1c4d3e7a6b2d5c8e3f9a1b7c6d4e5f",
            "ffb1a2c3d4e5f67890123456789abcdef"
        ],
        "Unwrapped": ["SingleLineValue", "AnotherOne", "NoBreaksHere"]
    })

    arrays = [
        ["Simulation", "Simulation", "Input", "Input", "Output"],
        ["ID", "Date Generated", "Model Name", "Parameters", "Result Summary"]
    ]
    columns = pd.MultiIndex.from_arrays(arrays)
    df5 = pd.DataFrame([
        [1, "2024-11-15", "RiskModelV2", "α=0.95, β=3.2", "Stable. 5 iterations. RMSE=0.003"],
        [2, "2025-02-04", "SuperModel", "α=0.99, β=2.1", "Converged quickly. RMSE=0.001"],
        [3, "2026-08-12", "LongModelNameWithDetails", "α=0.90, β=4.0, γ=1.0", "Diverged on step 4. RMSE=N/A"]
    ], columns=columns)

    return [df1, df2, df3, df4, df5]
