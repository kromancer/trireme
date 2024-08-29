import os
from pathlib import Path

import pandas as pd
import requests
from scipy.sparse import coo_matrix

from common import change_dir, read_config

url_base = "https://sparse.tamu.edu"


def get_index() -> pd.DataFrame:
    # URL of the CSV file containing the index of all available matrices
    url_csv = url_base + "/files/ssstats.csv"

    # Define the column headers
    column_headers = ["group", "name", "num_of_rows", "num_of_cols", "num_of_entries", "is_real", "is_binary",
                      "is_nd", "is_posdef", "psym", "nsym", "kind", "num_of_entries_redundant"]

    # Load the CSV file into a DataFrame, skipping the first two lines
    return pd.read_csv(url_csv, sep=",", skiprows=2, header=None, names=column_headers)


def get_all_suitesparse_matrices(dir: Path):
    df = get_index()
    with change_dir(dir):
        df.apply(lambda row: get_suitesparse_matrix(row["name"], row["group"]), axis=1)


def get_suitesparse_matrix(mtx_name: str, mtx_group: str = None):
    if mtx_group is None:
        df = get_index()
        mtx_group = df[df["name"] == mtx_name]["group"].values[0]

    # URL of the .tar.gz file
    filename = mtx_name + ".tar.gz"
    url_file = url_base + "/MM/" + mtx_group + "/" + mtx_name + ".tar.gz"

    response = requests.get(url_file, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.raw.read())
        print(f"Downloaded {filename}")
    else:
        raise RuntimeError(f"Failed to download {url_file}: {response.status_code}")
