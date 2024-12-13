import os
import lit.formats

config.name = "trireme"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.test', '.mlir']
config.test_source_root = os.path.dirname(__file__)

config.test_exec_root = os.path.join(config.test_source_root, ".lit")
config.excludes = [".lit"]

config.environment["PYTHONPATH"] = os.environ["PYTHONPATH"]
config.environment["LLVM_PATH"] = os.environ["LLVM_PATH"]

