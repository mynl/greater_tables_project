import pandas as pd
import numpy as np
import datetime as dt
from IPython.display import HTML, display
from . greater_tables import GT


# __gt_global = GT()


# def qhtml(df, **kwargs):
#     """Generic "quick display" function."""
#     return HTML(__gt_global(df, **kwargs))


# def qd(df, **kwargs):
#     """Generic "quick display" function."""
#     if isinstance(df, pd.Series):
#         if df.name is None:
#             df.name = 'value'
#         df = df.to_frame()
#     return display(HTML(__gt_global(df, **kwargs)))


def create_three_level_multiindex(df):
    """
    Adds two random levels to a DataFrame's column MultiIndex.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with a three-level MultiIndex on the columns.
    """
    n_columns = len(df.columns)
    level_1 = np.random.choice(["A", "B", "C"], size=n_columns)
    level_2 = np.random.choice(["X", "Y", "Z"], size=n_columns)

    # Create the MultiIndex
    multi_index = pd.MultiIndex.from_tuples(
        [(l1, l2, col) for l1, l2, col in zip(level_1, level_2, df.columns)],
        names=["Level 1", "Level 2", df.columns.name]
    )

    # Apply the new MultiIndex to the DataFrame
    df.columns = multi_index
    return df


def test_df(date=False, mi_columns=True):
    """Make a test dataframe nr rows with multi index."""
    nr = 10
    words = 'Parliament organised a year-long programme of events called Parliament in the Making to celebrate the 800th anniversary of the sealing of Magna Carta on 15 June and the 750th anniversary of the first representative parliament on 20 January Events were coordinated with Parliament Week'
    words = list(set(words.split(' ')))
    w1 = ['Abel', 'Cain', 'Issac', 'Fred', 'George']
    w2 = ['South', 'East', 'West', 'North']
    w4 = ['A', 'B', 'C', 'D']
    rints = [0, -10, 10, 100, -100, 1000, -1000, 10000, -10000, 100000, -100000, 1000000, -1000000]
    df = pd.DataFrame({'idx1': np.random.choice(w1, nr),
                       'idx2': np.random.choice(w2, nr),
                       'idx3': np.random.poisson(2, nr),
                       'floats': np.random.rand(nr) * 3000.,
                       'smaller': np.random.rand(nr) * 10 ** np.linspace(-3, 4, nr),
                       'larger': np.random.choice([-1., 0, 1.], nr) * np.random.rand(nr) * 10 ** np.linspace(3, 12, nr),
                       'ints': np.random.poisson(20, nr),
                       'powers': np.pi * 10. ** np.arange(-20, 26, 5),
                       'ratios': np.random.rand(nr) * 3. - 1.,
                       'string': [' '.join(np.random.choice(words, 4, replace=False)) for i in range(nr)],
                       # 'object': [np.random.poisson(2, nr) for i in range(nr)]
                       })

    if date:
        df['date'] = [dt.datetime.fromordinal(np.random.randint(dt.date(2020, 1, 1).toordinal(),
                                                                dt.date(2030, 1, 1).toordinal())) + dt.timedelta(seconds=np.random.randint(86400)) for _ in range(nr)]
        df['date'] = pd.to_datetime(df['date'])
    df.columns.name = 'Col name'
    df = df.set_index(['idx1', 'idx2', 'idx3'])
    if mi_columns:
        df = create_three_level_multiindex(df)

    # check unique and sort
    df = df.loc[df.index[~df.index.duplicated()]]
    assert np.all(~df.columns.duplicated()), 'Columns not all unique'
    df = df.sort_index(axis=0).sort_index(axis=1)
    return df


def make_test_dfs():
    """Make a dict of test dataframes with different characteristics."""
    ans = {}
    df = pd.DataFrame({'x': [int(i) for i in 3. ** np.arange(10)], 'y': np.arange(10, dtype=float)})
    df['w'] = df.x ** 0.35
    df['z'] = df.y ** .25

    ans['basic'] = df.copy()
    df1 = df.copy()
    df1.index.name = 'idx name'
    ans['basic w idx name'] = df1.copy()

    df1 = df.copy()
    df1.columns.name = 'col name'
    ans['basic w col name'] = df1.copy()

    df1 = df.copy()
    df1.index.name = 'idx name'
    df1.columns.name = 'col name'
    ans['basic w both names'] = df1.copy()

    df1['date'] = [dt.datetime.fromordinal(np.random.randint(dt.date(2020, 1, 1).toordinal(),
                   dt.date(2030, 1, 1).toordinal())) + dt.timedelta(seconds=np.random.randint(86400))
                    for _ in range(len(df1))]
    df2 = df1.set_index('date')
    ans['time series'] = df2.copy()

    df2 = df1.set_index('date', append=True)
    ans['time series and range'] = df2.copy()

    ans['realistic'] = test_df(date=False, mi_columns=False)
    ans['realistic w date'] = test_df(date=True, mi_columns=False).droplevel(2, axis=0)
    ans['realistic mi'] = test_df(date=False, mi_columns=True).droplevel(2, axis=1)
    ans['realistic mi w date'] = test_df(date=True, mi_columns=True).droplevel(2, axis=0).droplevel(2, axis=1)

    return ans
