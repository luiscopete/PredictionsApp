import numpy as np
import pandas as pd


def remove_outliers(data: pd.DataFrame) -> pd.DataFrame:
    """ Remove outliers from a list of numbers """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    step = 1.5 * (Q3 - Q1)
    # Remove outliers
    outliers_removed = [x for x in data if x >= Q1 - step and x <= Q3 + step]
    return outliers_removed
