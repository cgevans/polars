use crate::prelude::DecimalChunked;
use crate::utils::align_chunks_binary;
use super::*;

impl DecimalChunked {
    fn comparison_helper<Kernel, ScalarKernelLhs, ScalarKernelRhs>(
        &self,
        rhs: &DecimalChunked,
        kernel: Kernel,
        operation_lhs: ScalarKernelLhs,
        operation_rhs: ScalarKernelRhs,
    ) -> PolarsResult<BooleanChunked>
    where
        Kernel: Fn(&PrimitiveArray<i128>, &PrimitiveArray<i128>) -> BooleanArray,
        ScalarKernelLhs: Fn(&PrimitiveArray<i128>, i128) -> BooleanArray,
        ScalarKernelRhs: Fn(i128, &PrimitiveArray<i128>) -> BooleanArray,
    {
        let lhs = self;

        match (lhs.dtype(), rhs.dtype()) {
            (DataType::Decimal(p1, s1), DataType::Decimal(p2, s2)) => {
                if p1 != p2 || s1 != s2 {
                    polars_bail!(ComputeError: "Cannot compare decimals with different precision or scale")
                }
            },
            _ => polars_bail!(ComputeError: "Cannot compare arrays of different types"),
        }

        let mut ca = match (lhs.len(), rhs.len()) {
            (a, b) if a == b => {
                let (lhs, rhs) = align_chunks_binary(lhs, rhs);
                let iter = lhs
                    .downcast_iter()
                    .zip(rhs.downcast_iter())
                    .map(|(lhs, rhs)| kernel(lhs, rhs));
                ChunkedArray::from_chunk_iter(lhs.name(), iter)
            },
            // broadcast right path
            (_, 1) => {
                let opt_rhs = rhs.get(0);
                match opt_rhs {
                    None => ChunkedArray::full_null(lhs.name(), lhs.len()),
                    Some(rhs_val) => {
                        let iter = lhs.downcast_iter().map(|lhs| operation_lhs(lhs, rhs_val));
                        ChunkedArray::from_chunk_iter(lhs.name(), iter)
                    },
                }
            },
            (1, _) => {
                let opt_lhs = lhs.get(0);
                match opt_lhs {
                    None => ChunkedArray::full_null(lhs.name(), rhs.len()),
                    Some(lhs_val) => {
                        let iter = rhs.downcast_iter().map(|rhs| operation_rhs(lhs_val, rhs));
                        ChunkedArray::from_chunk_iter(lhs.name(), iter)
                    },
                }
            },
            _ => {
                polars_bail!(ComputeError: "Cannot apply operation on arrays of different lengths")
            },
        };
        ca.rename(lhs.name());
        Ok(ca)
    }
}

impl ChunkCompare<&DecimalChunked> for DecimalChunked {
    type Item = PolarsResult<BooleanChunked>;

    fn equal(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_eq_kernel(&rhs).into(),
            |arr, rhs| arr.tot_eq_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_eq_kernel_broadcast(&lhs).into(),
        )
    }

    fn equal_missing(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_eq_missing_kernel(&rhs).into(),
            |arr, rhs| arr.tot_eq_missing_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_eq_missing_kernel_broadcast(&lhs).into(),
        )
    }

    fn not_equal(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_ne_kernel(&rhs).into(),
            |arr, rhs| arr.tot_ne_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_ne_kernel_broadcast(&lhs).into(),
        )
    }

    fn not_equal_missing(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_ne_missing_kernel(&rhs).into(),
            |arr, rhs| arr.tot_ne_missing_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_ne_missing_kernel_broadcast(&lhs).into(),
        )
    }

    fn gt(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_gt_kernel(&rhs).into(),
            |arr, rhs| arr.tot_gt_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_gt_kernel_broadcast(&lhs).into(),
        )
    }

    fn gt_eq(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_ge_kernel(&rhs).into(),
            |arr, rhs| arr.tot_ge_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_ge_kernel_broadcast(&lhs).into(),
        )
    }

    fn lt(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_lt_kernel(&rhs).into(),
            |arr, rhs| arr.tot_lt_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_lt_kernel_broadcast(&lhs).into(),
        )
    }

    fn lt_eq(&self, rhs: &DecimalChunked) -> Self::Item {
        self.comparison_helper(
            rhs,
            |arr, rhs: &PrimitiveArray<i128>| arr.tot_le_kernel(&rhs).into(),
            |arr, rhs| arr.tot_le_kernel_broadcast(&rhs).into(),
            |lhs, arr| arr.tot_le_kernel_broadcast(&lhs).into(),
        )
    }
}
