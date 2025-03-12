from mlir import ir

with open("./gemm.mlir", "r") as f:
    gemm = f.read()

with ir.Context() as ctx:
    module = ir.Module.parse(gemm)

    def walk_ops(ops, when, what):
        for op in ops:
            if op.name == when:
                what(op)
            for region in op.regions:
                for block in region.blocks:
                    walk_ops(block.operations, when, what)

    def print_affine_maps(op):
        assert op.name == "linalg.generic"
        for attr in op.attributes:
            if attr.name == "indexing_maps":
                print(attr)


    walk_ops(module.body.operations, when="linalg.generic", what=print_affine_maps)



