from pathlib import Path
from time import time
from typing import Dict, Set

import pandas as pd
import requests

from common import change_dir
from singleton import Singleton

url_base = "https://sparse.tamu.edu"


class SuiteSparse(metaclass=Singleton):
    def __init__(self, working_dir: Path):
        self.dir = working_dir

        url_csv = url_base + "/files/ssstats.csv"
        self.column_headers = [
            "group", "name", "num_of_rows", "num_of_cols", "num_of_entries", "is_real", "is_binary",
            "is_nd", "is_posdef", "psym", "nsym", "kind", "num_of_entries_redundant"
        ]

        index_file = self.dir / "ssstats.csv"
        if index_file.exists():
            age = time() - index_file.stat().st_mtime
            if age < 432000:  # 43200 seconds = 5 days
                self.df = pd.read_csv(index_file, sep=",", skiprows=1, header=None, names=self.column_headers)
                return

        df = pd.read_csv(url_csv, sep=",", skiprows=2, header=None, names=self.column_headers)
        df.to_csv(index_file, index=False)
        self.df = df

    def get_info_url(self, mtx_name: str):
        return url_base + "/" + self.get_meta(mtx_name, "group") + "/" + mtx_name

    def get_meta(self, mtx_name: str, meta: str):
        assert meta in self.column_headers
        return self.df[self.df["name"] == mtx_name][meta].values[0]

    def is_binary(self, mtx_name: str) -> int:
        return self.get_meta(mtx_name, "is_binary")

    def is_pattern_symmetric(self, mtx_name: str) -> int:
        return self.get_meta(mtx_name, "psym")

    def get_all_matrix_names(self, is_real: bool = True) -> Set[str]:
        # filters non-complex matrices, including binary
        if is_real:
            df = self.df[self.df["is_real"] == 1]
        else:
            df = self.df

        return set(df["name"].values.tolist())

    def get_all_matrix_names_with_nnz(self) -> Dict[str, int]:
        return dict(zip(self.df["name"], self.df["num_of_entries"]))

    def get_all_matrices(self):
        with change_dir(self.dir):
            self.df.apply(lambda row: self.get_matrix(row["name"], row["group"]), axis=1)

    def get_matrix(self, mtx_name: str, mtx_group: str = None):
        if mtx_group is None:
            try:
                mtx_group = self.get_meta(mtx_name, "group")
            except IndexError:
                print(f"There is no SuiteSparse named {mtx_name}")
                exit(1)

        # URL of the .tar.gz file
        filename = mtx_name + ".tar.gz"
        url_file = url_base + "/RB/" + mtx_group + "/" + mtx_name + ".tar.gz"

        response = requests.get(url_file, stream=True)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.raw.read())
            print(f"Downloaded {filename}")
        else:
            raise RuntimeError(f"Failed to download {url_file}: {response.status_code}")

