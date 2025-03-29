from argparse import ArgumentParser, Namespace
from datetime import datetime
import json
from pathlib import Path
import sys
from _socket import gethostname
from statistics import mean, stdev
from git import Repo
from typing import List


class ReportManager:
    """
    Abstract base class for ReportManager to implement both real and no-op behavior.
    """

    @staticmethod
    def add_args(parser: ArgumentParser):
        def report_json(val: str) -> str:
            if not val.endswith(".json"):
                return val + ".json"
            return val

        parser.add_argument("--rep-file", type=report_json, default=None)

    @staticmethod
    def get_git_commit_hash():
        repo_path = Path(__file__).resolve().parent
        repo = Repo(repo_path, search_parent_directories=True)
        commit_hash = repo.head.commit.hexsha
        return commit_hash

    @staticmethod
    def get_stats(etimes_ns: List[int]):
        m = round(mean(etimes_ns) / 1000000, 3)
        std_dev = round(stdev(etimes_ns) / 1000000, 3)
        cv = round(std_dev / m, 3) if m != 0 else 0
        return m, std_dev, cv

    def append_placeholder(self, args: str = None):
        pass

    def append_result(self, new_entry, status="complete"):
        pass

    def log_execution_times_ns(self, etimes_ns: List[int]):
        pass

    def log_execution_times_secs(self, etimes_s: List[float]):
        self.log_execution_times_ns([int(t * 1e9) for t in etimes_s])


def create_report_manager(args: Namespace) -> ReportManager:
    """
    Factory function to create either a ReportManager or a no-op implementation.
    """
    if args.rep_file is None:
        return ReportManagerStdout()
    return DefaultReportManager(args.rep_file)


class ReportManagerStdout(ReportManager):
    """
    No-op implementation of ReportManager. Does nothing.
    """

    def __init__(self):
        pass

    def log_execution_times_ns(self, etimes_ns: List[int]):
        m, std_dev, cv = self.get_stats(etimes_ns)
        print(f"mean execution time: {m} ms")
        print(f"std dev: {std_dev} ms, CV: {cv * 100} %")


class DefaultReportManager(ReportManager):
    """
    Real implementation of the ReportManager that performs file operations.
    """

    def __init__(self, rep_file: str):
        self.rep_file = Path(rep_file).resolve()
        self.append_placeholder()

    def append_placeholder(self, args: str = None):
        try:
            with open(self.rep_file, 'r') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []
            print(f"Creating: {self.rep_file}")
        if not isinstance(data, list):
            data = []
        placeholder = {
            "args": " ".join(sys.argv) if args is None else args,
            "time": str(datetime.now()),
            "host": gethostname(),
            "git-hash": self.get_git_commit_hash(),
            "status": "initializing"
        }
        data.append(placeholder)
        with open(self.rep_file, 'w') as f:
            f.write(json.dumps(data, indent=4))

    def append_result(self, new_entry, status="complete"):
        with open(self.rep_file, 'r') as file:
            data = json.load(file)
        data[-1]["status"] = status
        data[-1].update(new_entry)
        with open(self.rep_file, 'w') as f:
            f.write(json.dumps(data, indent=4))

    def log_execution_times_ns(self, etimes_ns: List[int]):
        m, std_dev, cv = self.get_stats(etimes_ns)
        print(f"mean execution time: {m} ms")
        print(f"std dev: {std_dev} ms, CV: {cv * 100} %")
        self.append_result({
            'exec_times_ns': etimes_ns,
            'mean_ms': m,
            'std_dev': std_dev,
            'cv%': cv * 100})
