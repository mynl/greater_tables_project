"""
A massively over-engineered hash function for DataFrames. Output
varies with time. From GTP. Blake2b - who knew?
"""

import hashlib
import time
import base64
import pandas as pd

def df_short_hash(df, length=12):
    """Generate a short, time-dependent hash for a DataFrame (safe for HTML IDs)."""
    hasher = hashlib.blake2b(digest_size=8)  # Smaller output

    # Hash DataFrame content (values, index, columns)
    hasher.update(df.to_numpy().tobytes())
    hasher.update(pd.util.hash_pandas_object(df.index, index=True).values.tobytes())
    hasher.update(pd.util.hash_pandas_object(df.columns, index=True).values.tobytes())

    # Time component for uniqueness
    timestamp = str(time.time_ns()).encode()
    hasher.update(timestamp)

    # Encode as base32 (safe for HTML IDs, removes special characters)
    hash_bytes = hasher.digest()
    hash_str = base64.b32encode(hash_bytes).decode("utf-8").rstrip("=")  # Trim padding

    return f"T{hash_str[:length]}"  # Prefix with 'T' to ensure a valid ID
