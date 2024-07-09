import lit.formats
from pathlib import Path

config.name = "SparsificationPrefetches"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir']
config.test_source_root = Path(__file__).parent.resolve()
config.test_exec_root = Path.joinpath(config.test_source_root, ".lit")
