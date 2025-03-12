module {
    func.func @_internal_spmv(%0: i64) -> i32 {
        %16 = builtin.unrealized_conversion_cast %0 : i64 to index
        %17 = builtin.unrealized_conversion_cast %16 : index to i32
        return %17 : i32
    }
}
