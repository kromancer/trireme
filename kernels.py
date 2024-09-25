from mlir.dialects.linalg.opdsl import lang as dsl


@dsl.linalg_structured_op
def spvv_dsl(
    b=dsl.TensorDef(dsl.T, dsl.S.I),
    c=dsl.TensorDef(dsl.T, dsl.S.I),
    a=dsl.TensorDef(dsl.T, output=True)
):
    a[None] += b[dsl.D.i] * c[dsl.D.i]


@dsl.linalg_structured_op
def spmv_dsl(
    B=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J),
    c=dsl.TensorDef(dsl.T, dsl.S.J),
    a=dsl.TensorDef(dsl.T, dsl.S.I, output=True)
):
    a[dsl.D.i] += B[dsl.D.i, dsl.D.j] * c[dsl.D.j]


@dsl.linalg_structured_op
def spmm_dsl(
    B=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.K),
    C=dsl.TensorDef(dsl.T, dsl.S.K, dsl.S.J),
    A=dsl.TensorDef(dsl.T, dsl.S.I, dsl.S.J, output=True)
):
    A[dsl.D.i, dsl.D.j] += B[dsl.D.i, dsl.D.k] * C[dsl.D.k, dsl.D.j]
