// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines take kernel for [Array]

use std::collections::HashMap;
use std::fmt::Display;
use std::mem::ManuallyDrop;
use std::sync::Arc;

use arrow_array::builder::{BufferBuilder, UInt32Builder};
use arrow_array::cast::AsArray;
use arrow_array::types::*;
use arrow_array::*;
use arrow_buffer::{
    ArrowNativeType, BooleanBuffer, Buffer, MutableBuffer, NullBuffer, OffsetBuffer, ScalarBuffer,
    bit_util,
};
use arrow_data::{
    ArrayData, ArrayDataBuilder, ByteView, MAX_INLINE_VIEW_LEN, transform::MutableArrayData,
};
use arrow_schema::{ArrowError, DataType, FieldRef, UnionMode};

use num_traits::Zero;

/// Take elements by index from [Array], creating a new [Array] from those indexes.
///
/// ```text
/// ┌─────────────────┐      ┌─────────┐                              ┌─────────────────┐
/// │        A        │      │    0    │                              │        A        │
/// ├─────────────────┤      ├─────────┤                              ├─────────────────┤
/// │        D        │      │    2    │                              │        B        │
/// ├─────────────────┤      ├─────────┤   take(values, indices)      ├─────────────────┤
/// │        B        │      │    3    │ ─────────────────────────▶   │        C        │
/// ├─────────────────┤      ├─────────┤                              ├─────────────────┤
/// │        C        │      │    1    │                              │        D        │
/// ├─────────────────┤      └─────────┘                              └─────────────────┘
/// │        E        │
/// └─────────────────┘
///    values array          indices array                              result
/// ```
///
/// For selecting values by index from multiple arrays see [`crate::interleave`]
///
/// Note that this kernel, similar to other kernels in this crate,
/// will avoid allocating where not necessary. Consequently
/// the returned array may share buffers with the inputs
///
/// # Errors
/// This function errors whenever:
/// * An index cannot be casted to `usize` (typically 32 bit architectures)
/// * An index is out of bounds and `options` is set to check bounds.
///
/// # Safety
///
/// When `options` is not set to check bounds, taking indexes after `len` will panic.
///
/// # See also
/// * [`BatchCoalescer`]: to filter multiple [`RecordBatch`] and coalesce
///   the results into a single array.
///
/// [`BatchCoalescer`]: crate::coalesce::BatchCoalescer
///
/// # Examples
/// ```
/// # use arrow_array::{StringArray, UInt32Array, cast::AsArray};
/// # use arrow_select::take::take;
/// let values = StringArray::from(vec!["zero", "one", "two"]);
///
/// // Take items at index 2, and 1:
/// let indices = UInt32Array::from(vec![2, 1]);
/// let taken = take(&values, &indices, None).unwrap();
/// let taken = taken.as_string::<i32>();
///
/// assert_eq!(*taken, StringArray::from(vec!["two", "one"]));
/// ```
pub fn take(
    values: &dyn Array,
    indices: &dyn Array,
    options: Option<TakeOptions>,
) -> Result<ArrayRef, ArrowError> {
    let options = options.unwrap_or_default();
    downcast_integer_array!(
        indices => {
            if options.check_bounds {
                check_bounds(values.len(), indices)?;
            }
            let indices = indices.to_indices();
            take_impl(values, &indices).map(deep_compact_views)
        },
        d => Err(ArrowError::InvalidArgumentError(format!("Take only supported for integers, got {d:?}")))
    )
}

/// For each [ArrayRef] in the [`Vec<ArrayRef>`], take elements by index and create a new
/// [`Vec<ArrayRef>`] from those indices.
///
/// ```text
/// ┌────────┬────────┐
/// │        │        │           ┌────────┐                                ┌────────┬────────┐
/// │   A    │   1    │           │        │                                │        │        │
/// ├────────┼────────┤           │   0    │                                │   A    │   1    │
/// │        │        │           ├────────┤                                ├────────┼────────┤
/// │   D    │   4    │           │        │                                │        │        │
/// ├────────┼────────┤           │   2    │  take_arrays(values,indices)   │   B    │   2    │
/// │        │        │           ├────────┤                                ├────────┼────────┤
/// │   B    │   2    │           │        │  ───────────────────────────►  │        │        │
/// ├────────┼────────┤           │   3    │                                │   C    │   3    │
/// │        │        │           ├────────┤                                ├────────┼────────┤
/// │   C    │   3    │           │        │                                │        │        │
/// ├────────┼────────┤           │   1    │                                │   D    │   4    │
/// │        │        │           └────────┘                                └────────┼────────┘
/// │   E    │   5    │
/// └────────┴────────┘
///    values arrays             indices array                                      result
/// ```
///
/// # Errors
/// This function errors whenever:
/// * An index cannot be casted to `usize` (typically 32 bit architectures)
/// * An index is out of bounds and `options` is set to check bounds.
///
/// # Safety
///
/// When `options` is not set to check bounds, taking indexes after `len` will panic.
///
/// # Examples
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::{StringArray, UInt32Array, cast::AsArray};
/// # use arrow_select::take::{take, take_arrays};
/// let string_values = Arc::new(StringArray::from(vec!["zero", "one", "two"]));
/// let values = Arc::new(UInt32Array::from(vec![0, 1, 2]));
///
/// // Take items at index 2, and 1:
/// let indices = UInt32Array::from(vec![2, 1]);
/// let taken_arrays = take_arrays(&[string_values, values], &indices, None).unwrap();
/// let taken_string = taken_arrays[0].as_string::<i32>();
/// assert_eq!(*taken_string, StringArray::from(vec!["two", "one"]));
/// let taken_values = taken_arrays[1].as_primitive();
/// assert_eq!(*taken_values, UInt32Array::from(vec![2, 1]));
/// ```
pub fn take_arrays(
    arrays: &[ArrayRef],
    indices: &dyn Array,
    options: Option<TakeOptions>,
) -> Result<Vec<ArrayRef>, ArrowError> {
    arrays
        .iter()
        .map(|array| take(array.as_ref(), indices, options.clone()))
        .collect()
}

/// Verifies that the non-null values of `indices` are all `< len`
fn check_bounds<T: ArrowPrimitiveType>(
    len: usize,
    indices: &PrimitiveArray<T>,
) -> Result<(), ArrowError>
where
    T::Native: Display,
{
    let len = match T::Native::from_usize(len) {
        Some(len) => len,
        None => {
            if T::DATA_TYPE.is_integer() {
                // the biggest representable value for T::Native is lower than len, e.g: u8::MAX < 512, no need to check bounds
                return Ok(());
            } else {
                return Err(ArrowError::ComputeError("Cast to usize failed".to_string()));
            }
        }
    };

    if indices.null_count() > 0 {
        indices.iter().flatten().try_for_each(|index| {
            if index >= len {
                return Err(ArrowError::ComputeError(format!(
                    "Array index out of bounds, cannot get item at index {index} from {len} entries"
                )));
            }
            Ok(())
        })
    } else {
        let in_bounds = indices.values().iter().fold(true, |in_bounds, &i| {
            in_bounds & (i >= T::Native::ZERO) & (i < len)
        });

        if !in_bounds {
            for &index in indices.values() {
                if index < T::Native::ZERO || index >= len {
                    return Err(ArrowError::ComputeError(format!(
                        "Array index out of bounds, cannot get item at index {index} from {len} entries"
                    )));
                }
            }
        }

        Ok(())
    }
}

#[inline(never)]
fn take_impl<IndexType: ArrowPrimitiveType>(
    values: &dyn Array,
    indices: &PrimitiveArray<IndexType>,
) -> Result<ArrayRef, ArrowError> {
    if indices.is_empty() {
        return Ok(new_empty_array(values.data_type()));
    }
    downcast_primitive_array! {
        values => Ok(Arc::new(take_primitive(values, indices)?)),
        DataType::Boolean => {
            let values = values.as_any().downcast_ref::<BooleanArray>().unwrap();
            Ok(Arc::new(take_boolean(values, indices)))
        }
        DataType::Utf8 => {
            Ok(Arc::new(take_bytes(values.as_string::<i32>(), indices)?))
        }
        DataType::LargeUtf8 => {
            Ok(Arc::new(take_bytes(values.as_string::<i64>(), indices)?))
        }
        DataType::Utf8View => {
            Ok(Arc::new(take_byte_view(values.as_string_view(), indices)?))
        }
        DataType::List(_) => {
            Ok(Arc::new(take_list::<_, Int32Type>(values.as_list(), indices)?))
        }
        DataType::LargeList(_) => {
            Ok(Arc::new(take_list::<_, Int64Type>(values.as_list(), indices)?))
        }
        DataType::ListView(_) => {
            Ok(Arc::new(take_list_view::<_, Int32Type>(values.as_list_view(), indices)?))
        }
        DataType::LargeListView(_) => {
            Ok(Arc::new(take_list_view::<_, Int64Type>(values.as_list_view(), indices)?))
        }
        DataType::FixedSizeList(_, length) => {
            let values = values
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap();
            Ok(Arc::new(take_fixed_size_list(
                values,
                indices,
                *length as u32,
            )?))
        }
        DataType::Map(_, _) => {
            let list_arr = ListArray::from(values.as_map().clone());
            let list_data = take_list::<_, Int32Type>(&list_arr, indices)?;
            let builder = list_data.into_data().into_builder().data_type(values.data_type().clone());
            Ok(Arc::new(MapArray::from(unsafe { builder.build_unchecked() })))
        }
        DataType::Struct(fields) => {
            let array: &StructArray = values.as_struct();
            let arrays  = array
                .columns()
                .iter()
                .map(|a| take_impl(a.as_ref(), indices))
                .collect::<Result<Vec<ArrayRef>, _>>()?;
            let fields: Vec<(FieldRef, ArrayRef)> =
                fields.iter().cloned().zip(arrays).collect();

            // Create the null bit buffer.
            let is_valid: Buffer = indices
                .iter()
                .map(|index| {
                    if let Some(index) = index {
                        array.is_valid(index.to_usize().unwrap())
                    } else {
                        false
                    }
                })
                .collect();

            if fields.is_empty() {
                let nulls = NullBuffer::new(BooleanBuffer::new(is_valid, 0, indices.len()));
                Ok(Arc::new(StructArray::new_empty_fields(indices.len(), Some(nulls))))
            } else {
                Ok(Arc::new(StructArray::from((fields, is_valid))) as ArrayRef)
            }
        }
        DataType::Dictionary(_, _) => downcast_dictionary_array! {
            values => Ok(Arc::new(take_dict(values, indices)?)),
            t => unimplemented!("Take not supported for dictionary type {:?}", t)
        }
        DataType::RunEndEncoded(_, _) => downcast_run_array! {
            values => Ok(Arc::new(take_run(values, indices)?)),
            t => unimplemented!("Take not supported for run type {:?}", t)
        }
        DataType::Binary => {
            Ok(Arc::new(take_bytes(values.as_binary::<i32>(), indices)?))
        }
        DataType::LargeBinary => {
            Ok(Arc::new(take_bytes(values.as_binary::<i64>(), indices)?))
        }
        DataType::BinaryView => {
            Ok(Arc::new(take_byte_view(values.as_binary_view(), indices)?))
        }
        DataType::FixedSizeBinary(size) => {
            let values = values
                .as_any()
                .downcast_ref::<FixedSizeBinaryArray>()
                .unwrap();
            Ok(Arc::new(take_fixed_size_binary(values, indices, *size)?))
        }
        DataType::Null => {
            // Take applied to a null array produces a null array.
            if values.len() >= indices.len() {
                // If the existing null array is as big as the indices, we can use a slice of it
                // to avoid allocating a new null array.
                Ok(values.slice(0, indices.len()))
            } else {
                // If the existing null array isn't big enough, create a new one.
                Ok(new_null_array(&DataType::Null, indices.len()))
            }
        }
        DataType::Union(fields, UnionMode::Sparse) => {
            let mut children = Vec::with_capacity(fields.len());
            let values = values.as_any().downcast_ref::<UnionArray>().unwrap();
            let type_ids = take_native(values.type_ids(), indices);
            for (type_id, _field) in fields.iter() {
                let values = values.child(type_id);
                let values = take_impl(values, indices)?;
                children.push(values);
            }
            let array = UnionArray::try_new(fields.clone(), type_ids, None, children)?;
            Ok(Arc::new(array))
        }
        DataType::Union(fields, UnionMode::Dense) => {
            let values = values.as_any().downcast_ref::<UnionArray>().unwrap();

            let type_ids = <PrimitiveArray<Int8Type>>::try_new(take_native(values.type_ids(), indices), None)?;
            let offsets = <PrimitiveArray<Int32Type>>::try_new(take_native(values.offsets().unwrap(), indices), None)?;

            let children = fields.iter()
                .map(|(field_type_id, _)| {
                    let mask = BooleanArray::from_unary(&type_ids, |value_type_id| value_type_id == field_type_id);

                    let indices = crate::filter::filter(&offsets, &mask)?;

                    let values = values.child(field_type_id);

                    take_impl(values, indices.as_primitive::<Int32Type>())
                })
                .collect::<Result<_, _>>()?;

            let mut child_offsets = [0; 128];

            let offsets = type_ids.values()
                .iter()
                .map(|&i| {
                    let offset = child_offsets[i as usize];

                    child_offsets[i as usize] += 1;

                    offset
                })
                .collect();

            let (_, type_ids, _) = type_ids.into_parts();

            let array = UnionArray::try_new(fields.clone(), type_ids, Some(offsets), children)?;

            Ok(Arc::new(array))
        }
        t => unimplemented!("Take not supported for data type {:?}", t)
    }
}

/// Options that define how `take` should behave
#[derive(Clone, Debug, Default)]
pub struct TakeOptions {
    /// Perform bounds check before taking indices from values.
    /// If enabled, an `ArrowError` is returned if the indices are out of bounds.
    /// If not enabled, and indices exceed bounds, the kernel will panic.
    pub check_bounds: bool,
}

/// `take` implementation for all primitive arrays
///
/// This checks if an `indices` slot is populated, and gets the value from `values`
///  as the populated index.
/// If the `indices` slot is null, a null value is returned.
/// For example, given:
///     values:  [1, 2, 3, null, 5]
///     indices: [0, null, 4, 3]
/// The result is: [1 (slot 0), null (null slot), 5 (slot 4), null (slot 3)]
fn take_primitive<T, I>(
    values: &PrimitiveArray<T>,
    indices: &PrimitiveArray<I>,
) -> Result<PrimitiveArray<T>, ArrowError>
where
    T: ArrowPrimitiveType,
    I: ArrowPrimitiveType,
{
    let values_buf = take_native(values.values(), indices);
    let nulls = take_nulls(values.nulls(), indices);
    Ok(PrimitiveArray::try_new(values_buf, nulls)?.with_data_type(values.data_type().clone()))
}

#[inline(never)]
fn take_nulls<I: ArrowPrimitiveType>(
    values: Option<&NullBuffer>,
    indices: &PrimitiveArray<I>,
) -> Option<NullBuffer> {
    match values.filter(|n| n.null_count() > 0) {
        Some(n) => NullBuffer::from_unsliced_buffer(
            take_bits(n.inner(), indices).into_inner(),
            indices.len(),
        ),
        None => indices.nulls().cloned(),
    }
}

#[inline(never)]
fn take_native<T: ArrowNativeType, I: ArrowPrimitiveType>(
    values: &[T],
    indices: &PrimitiveArray<I>,
) -> ScalarBuffer<T> {
    match indices.nulls().filter(|n| n.null_count() > 0) {
        Some(n) => indices
            .values()
            .iter()
            .enumerate()
            .map(|(idx, index)| match values.get(index.as_usize()) {
                Some(v) => *v,
                // SAFETY: idx<indices.len()
                None => match unsafe { n.inner().value_unchecked(idx) } {
                    false => T::default(),
                    true => panic!("Out-of-bounds index {index:?}"),
                },
            })
            .collect(),
        None => indices
            .values()
            .iter()
            .map(|index| values[index.as_usize()])
            .collect(),
    }
}

#[inline(never)]
fn take_bits<I: ArrowPrimitiveType>(
    values: &BooleanBuffer,
    indices: &PrimitiveArray<I>,
) -> BooleanBuffer {
    let len = indices.len();

    match indices.nulls().filter(|n| n.null_count() > 0) {
        Some(nulls) => {
            let mut output_buffer = MutableBuffer::new_null(len);
            let output_slice = output_buffer.as_slice_mut();
            nulls.valid_indices().for_each(|idx| {
                // SAFETY: idx is a valid index in indices.nulls() --> idx<indices.len()
                if values.value(unsafe { indices.value_unchecked(idx).as_usize() }) {
                    // SAFETY: MutableBuffer was created with space for indices.len() bit, and idx < indices.len()
                    unsafe { bit_util::set_bit_raw(output_slice.as_mut_ptr(), idx) };
                }
            });
            BooleanBuffer::new(output_buffer.into(), 0, len)
        }
        None => {
            BooleanBuffer::collect_bool(len, |idx: usize| {
                // SAFETY: idx<indices.len()
                values.value(unsafe { indices.value_unchecked(idx).as_usize() })
            })
        }
    }
}

/// `take` implementation for boolean arrays
fn take_boolean<IndexType: ArrowPrimitiveType>(
    values: &BooleanArray,
    indices: &PrimitiveArray<IndexType>,
) -> BooleanArray {
    let val_buf = take_bits(values.values(), indices);
    let null_buf = take_nulls(values.nulls(), indices);
    BooleanArray::new(val_buf, null_buf)
}

/// `take` implementation for string arrays
fn take_bytes<T: ByteArrayType, IndexType: ArrowPrimitiveType>(
    array: &GenericByteArray<T>,
    indices: &PrimitiveArray<IndexType>,
) -> Result<GenericByteArray<T>, ArrowError> {
    let mut offsets = Vec::with_capacity(indices.len() + 1);
    offsets.push(T::Offset::default());

    let input_offsets = array.value_offsets();
    let mut capacity = 0;
    let nulls = take_nulls(array.nulls(), indices);

    let (offsets, values) = if array.null_count() == 0 && indices.null_count() == 0 {
        offsets.reserve(indices.len());
        for index in indices.values() {
            let index = index.as_usize();
            capacity += input_offsets[index + 1].as_usize() - input_offsets[index].as_usize();
            offsets.push(
                T::Offset::from_usize(capacity)
                    .ok_or_else(|| ArrowError::OffsetOverflowError(capacity))?,
            );
        }
        let mut values = Vec::with_capacity(capacity);

        for index in indices.values() {
            values.extend_from_slice(array.value(index.as_usize()).as_ref());
        }
        (offsets, values)
    } else if indices.null_count() == 0 {
        offsets.reserve(indices.len());
        for index in indices.values() {
            let index = index.as_usize();
            if array.is_valid(index) {
                capacity += input_offsets[index + 1].as_usize() - input_offsets[index].as_usize();
            }
            offsets.push(
                T::Offset::from_usize(capacity)
                    .ok_or_else(|| ArrowError::OffsetOverflowError(capacity))?,
            );
        }
        let mut values = Vec::with_capacity(capacity);

        for index in indices.values() {
            let index = index.as_usize();
            if array.is_valid(index) {
                values.extend_from_slice(array.value(index).as_ref());
            }
        }
        (offsets, values)
    } else if array.null_count() == 0 {
        offsets.reserve(indices.len());
        for (i, index) in indices.values().iter().enumerate() {
            let index = index.as_usize();
            if indices.is_valid(i) {
                capacity += input_offsets[index + 1].as_usize() - input_offsets[index].as_usize();
            }
            offsets.push(
                T::Offset::from_usize(capacity)
                    .ok_or_else(|| ArrowError::OffsetOverflowError(capacity))?,
            );
        }
        let mut values = Vec::with_capacity(capacity);

        for (i, index) in indices.values().iter().enumerate() {
            if indices.is_valid(i) {
                values.extend_from_slice(array.value(index.as_usize()).as_ref());
            }
        }
        (offsets, values)
    } else {
        let nulls = nulls.as_ref().unwrap();
        offsets.reserve(indices.len());
        for (i, index) in indices.values().iter().enumerate() {
            let index = index.as_usize();
            if nulls.is_valid(i) {
                capacity += input_offsets[index + 1].as_usize() - input_offsets[index].as_usize();
            }
            offsets.push(
                T::Offset::from_usize(capacity)
                    .ok_or_else(|| ArrowError::OffsetOverflowError(capacity))?,
            );
        }
        let mut values = Vec::with_capacity(capacity);

        for (i, index) in indices.values().iter().enumerate() {
            // check index is valid before using index. The value in
            // NULL index slots may not be within bounds of array
            let index = index.as_usize();
            if nulls.is_valid(i) {
                values.extend_from_slice(array.value(index).as_ref());
            }
        }
        (offsets, values)
    };

    T::Offset::from_usize(values.len())
        .ok_or_else(|| ArrowError::OffsetOverflowError(values.len()))?;

    let array = unsafe {
        let offsets = OffsetBuffer::new_unchecked(offsets.into());
        GenericByteArray::<T>::new_unchecked(offsets, values.into(), nulls)
    };

    Ok(array)
}

/// `take` implementation for byte view arrays
fn take_byte_view<T: ByteViewType, IndexType: ArrowPrimitiveType>(
    array: &GenericByteViewArray<T>,
    indices: &PrimitiveArray<IndexType>,
) -> Result<GenericByteViewArray<T>, ArrowError> {
    let new_views = take_native(array.views(), indices);
    let new_nulls = take_nulls(array.nulls(), indices);
    // Safety:  array.views was valid, and take_native copies only valid values, and verifies bounds
    Ok(unsafe {
        GenericByteViewArray::new_unchecked(new_views, array.data_buffers().to_vec(), new_nulls)
    })
}

/// Compact every sparse view array in `array`, however deeply nested.
/// Returns the input unchanged (same `Arc`) when nothing needs compaction.
///
/// `take` keeps the input's entire data-buffer list regardless of how few
/// rows survive, and `concat` appends every input's list, so a chain of
/// row-reducing operators (e.g. hash joins) otherwise accumulates every
/// upstream buffer: the list grows multiplicatively per level and the
/// buffers stay alive. Applied at the end of the public `take` and `concat`
/// entry points.
///
/// Reachability is judged per view array from its own views; rows a parent
/// hides (list-view offsets, sliced containers) still count as referenced,
/// so such children are left alone. They share their child arrays with the
/// source, so their lists cannot grow through this path either.
///
/// Exposed for callers that retain a selection past the lifetime of the batch
/// it came from, such as a top-N heap or a hash-join build side. Such a caller
/// knows what the kernels cannot: that the output will outlive its source, so
/// compaction is worth paying for even when the kernels' own thresholds leave
/// the array alone.
pub fn deep_compact_views(array: ArrayRef) -> ArrayRef {
    if !type_contains_views(array.data_type()) {
        return array;
    }
    match deep_compact_data(&array.to_data()) {
        Some(data) => make_array(data),
        None => array,
    }
}

/// Recursive worker for [`deep_compact_views`]; `None` means unchanged.
fn deep_compact_data(data: &ArrayData) -> Option<ArrayData> {
    match data.data_type() {
        DataType::Utf8View => {
            let array = StringViewArray::from(data.clone());
            compact_byte_view(&array).map(|a| a.into_data())
        }
        DataType::BinaryView => {
            let array = BinaryViewArray::from(data.clone());
            compact_byte_view(&array).map(|a| a.into_data())
        }
        dt if type_contains_views(dt) => {
            let mut changed = false;
            let children: Vec<ArrayData> = data
                .child_data()
                .iter()
                .map(|child| match deep_compact_data(child) {
                    Some(new) => {
                        changed = true;
                        new
                    }
                    None => child.clone(),
                })
                .collect();
            if !changed {
                return None;
            }
            // SAFETY: same type, length, offset, buffers, and nulls as
            // `data`, which was valid; each child is replaced by a logically
            // identical compacted array of the same length.
            Some(unsafe {
                data.clone()
                    .into_builder()
                    .child_data(children)
                    .build_unchecked()
            })
        }
        _ => None,
    }
}

/// Threshold below which a view array is left alone: bounded retention
/// this small is cheaper than scanning the views, and buffer lists this
/// short cannot be mid-compounding (each level multiplies the list).
const COMPACT_MIN_CAPACITY: usize = 1024 * 1024;
const COMPACT_MAX_SKIP_BUFFERS: usize = 16;

/// Bound what a view array retains beyond what its views reference.
/// Returns `None` when the array is left unchanged.
///
/// Two stages. Unreferenced and duplicate buffers are dropped and the view
/// indices remapped, copying no data; this is what stops buffer lists from
/// compounding through chained `take`/`concat`. `gc()` then copies the
/// referenced bytes only when the kept buffers hold more than twice the
/// referenced bytes and the excess exceeds [`COMPACT_MIN_CAPACITY`], so
/// retention is bounded, not eliminated: an array can keep up to
/// `max(2x referenced, referenced + COMPACT_MIN_CAPACITY)` alive.
fn compact_byte_view<T: ByteViewType>(
    array: &GenericByteViewArray<T>,
) -> Option<GenericByteViewArray<T>> {
    let buffers = array.data_buffers();
    if buffers.is_empty() {
        return None;
    }
    // Cheap gate before the O(views) scan: small, short-listed arrays are
    // not worth touching (typical outputs of already-compact batches).
    let total_capacity: usize = buffers.iter().map(|b| b.capacity()).sum();
    if total_capacity <= COMPACT_MIN_CAPACITY && buffers.len() <= COMPACT_MAX_SKIP_BUFFERS {
        return None;
    }

    // One pass over the views: bytes referenced and which buffers are used.
    let mut used_bytes: usize = 0;
    let mut referenced = vec![false; buffers.len()];
    for v in array.views().iter() {
        let len = *v as u32;
        if len > MAX_INLINE_VIEW_LEN {
            let view = ByteView::from(*v);
            used_bytes += len as usize;
            // A view pointing outside the buffer list means the array
            // skipped validation; leave it alone rather than panic.
            match referenced.get_mut(view.buffer_index as usize) {
                Some(slot) => *slot = true,
                None => return None,
            }
        }
    }

    // Keep one entry per referenced allocation (`concat` clones the same
    // buffer once per input) and remember each old index's replacement.
    let mut kept: Vec<Buffer> = Vec::new();
    let mut kept_capacity: usize = 0;
    let mut by_alloc: HashMap<(usize, usize), u32> = HashMap::new();
    let mut remap: Vec<u32> = vec![u32::MAX; buffers.len()];
    for (i, buffer) in buffers.iter().enumerate() {
        if !referenced[i] {
            continue;
        }
        let key = (buffer.as_ptr() as usize, buffer.len());
        remap[i] = *by_alloc.entry(key).or_insert_with(|| {
            kept_capacity += buffer.capacity();
            kept.push(buffer.clone());
            (kept.len() - 1) as u32
        });
    }

    // Copy only when the excess is both relatively (2x) and absolutely
    // large; mid-selectivity takes stay zero-copy.
    //
    // NOTE: the decision deliberately ignores how many holders the kept
    // buffers have, even though a copy frees nothing unless this array is the
    // last holder. Inside a kernel the caller still owns the input, so such a
    // check could essentially never fire, and it would skip exactly the case
    // that matters: an output that is retained while its source batch is
    // dropped right after the call. Waste is the only signal available here.
    // A consumer calling `deep_compact_views` at the point it stores a batch
    // can make the sharper decision.
    let waste = kept_capacity.saturating_sub(used_bytes);
    if kept_capacity > used_bytes.saturating_mul(2) && waste > COMPACT_MIN_CAPACITY {
        return Some(array.gc());
    }
    if kept.len() == buffers.len() {
        return None;
    }

    let views: Vec<u128> = array
        .views()
        .iter()
        .map(|v| {
            let len = *v as u32;
            if len > MAX_INLINE_VIEW_LEN {
                let mut view = ByteView::from(*v);
                view.buffer_index = remap[view.buffer_index as usize];
                view.as_u128()
            } else {
                *v
            }
        })
        .collect();
    // SAFETY: every remapped index points at a clone of the buffer the view
    // pointed at before; inline views and nulls are unchanged.
    Some(unsafe { GenericByteViewArray::new_unchecked(views.into(), kept, array.nulls().cloned()) })
}

fn type_contains_views(dt: &DataType) -> bool {
    match dt {
        DataType::Utf8View | DataType::BinaryView => true,
        DataType::Struct(fields) => fields.iter().any(|f| type_contains_views(f.data_type())),
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => type_contains_views(f.data_type()),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, f)| type_contains_views(f.data_type())),
        // Dictionary values are shared wholesale by `take` (no list growth),
        // and RunEndEncoded values are taken through the public `take`,
        // which compacts them; neither needs an arm here.
        _ => false,
    }
}

/// `take` implementation for list arrays
///
/// Copies the selected list entries' child slices into a new child array
/// via `MutableArrayData`, then reconstructs a list array with new offsets
fn take_list<IndexType, OffsetType>(
    values: &GenericListArray<OffsetType::Native>,
    indices: &PrimitiveArray<IndexType>,
) -> Result<GenericListArray<OffsetType::Native>, ArrowError>
where
    IndexType: ArrowPrimitiveType,
    OffsetType: ArrowPrimitiveType,
    OffsetType::Native: OffsetSizeTrait,
    PrimitiveArray<OffsetType>: From<Vec<OffsetType::Native>>,
{
    let list_offsets = values.value_offsets();
    let child_data = values.values().to_data();
    let nulls = take_nulls(values.nulls(), indices);

    let mut new_offsets = Vec::with_capacity(indices.len() + 1);
    new_offsets.push(OffsetType::Native::zero());

    let use_nulls = child_data.null_count() > 0;

    let capacity = child_data
        .len()
        .checked_div(values.len())
        .map(|v| v * indices.len())
        .unwrap_or_default();

    let mut array_data = MutableArrayData::new(vec![&child_data], use_nulls, capacity);

    match nulls.as_ref().filter(|n| n.null_count() > 0) {
        None => {
            for index in indices.values() {
                let ix = index.as_usize();
                let start = list_offsets[ix].as_usize();
                let end = list_offsets[ix + 1].as_usize();
                array_data.extend(0, start, end);
                new_offsets.push(OffsetType::Native::from_usize(array_data.len()).unwrap());
            }
        }
        Some(output_nulls) => {
            assert_eq!(output_nulls.len(), indices.len());

            let mut last_filled = 0;
            for i in output_nulls.valid_indices() {
                let current = OffsetType::Native::from_usize(array_data.len()).unwrap();
                // Filling offsets for the null values between the two valid indices
                if last_filled < i {
                    new_offsets.extend(std::iter::repeat_n(current, i - last_filled));
                }

                // SAFETY: `i` comes from validity bitmap over `indices`, so in-bounds.
                let ix = unsafe { indices.value_unchecked(i) }.as_usize();
                let start = list_offsets[ix].as_usize();
                let end = list_offsets[ix + 1].as_usize();
                array_data.extend(0, start, end);
                new_offsets.push(OffsetType::Native::from_usize(array_data.len()).unwrap());
                last_filled = i + 1;
            }

            // Filling offsets for null values at the end
            let final_offset = OffsetType::Native::from_usize(array_data.len()).unwrap();
            new_offsets.extend(std::iter::repeat_n(
                final_offset,
                indices.len() - last_filled,
            ));
        }
    };

    assert_eq!(
        new_offsets.len(),
        indices.len() + 1,
        "New offsets was filled under/over the expected capacity"
    );

    let child_data = array_data.freeze();
    let value_offsets = Buffer::from_vec(new_offsets);

    let list_data = ArrayDataBuilder::new(values.data_type().clone())
        .len(indices.len())
        .nulls(nulls)
        .offset(0)
        .add_child_data(child_data)
        .add_buffer(value_offsets);

    let list_data = unsafe { list_data.build_unchecked() };
    Ok(GenericListArray::<OffsetType::Native>::from(list_data))
}

fn take_list_view<IndexType, OffsetType>(
    values: &GenericListViewArray<OffsetType::Native>,
    indices: &PrimitiveArray<IndexType>,
) -> Result<GenericListViewArray<OffsetType::Native>, ArrowError>
where
    IndexType: ArrowPrimitiveType,
    OffsetType: ArrowPrimitiveType,
    OffsetType::Native: OffsetSizeTrait,
{
    let taken_offsets = take_native(values.offsets(), indices);
    let taken_sizes = take_native(values.sizes(), indices);
    let nulls = take_nulls(values.nulls(), indices);

    let list_view_data = ArrayDataBuilder::new(values.data_type().clone())
        .len(indices.len())
        .nulls(nulls)
        .buffers(vec![taken_offsets.into(), taken_sizes.into()])
        .child_data(vec![values.values().to_data()]);

    // SAFETY: all buffers and child nodes for ListView added in constructor
    let list_view_data = unsafe { list_view_data.build_unchecked() };

    Ok(GenericListViewArray::<OffsetType::Native>::from(
        list_view_data,
    ))
}

/// `take` implementation for `FixedSizeListArray`
///
/// Calculates the index and indexed offset for the inner array,
/// applying `take` on the inner array, then reconstructing a list array
/// with the indexed offsets
fn take_fixed_size_list<IndexType: ArrowPrimitiveType>(
    values: &FixedSizeListArray,
    indices: &PrimitiveArray<IndexType>,
    length: <UInt32Type as ArrowPrimitiveType>::Native,
) -> Result<FixedSizeListArray, ArrowError> {
    let list_indices = take_value_indices_from_fixed_size_list(values, indices, length)?;
    let taken = take_impl::<UInt32Type>(values.values().as_ref(), &list_indices)?;

    // determine null count and null buffer, which are a function of `values` and `indices`
    let num_bytes = bit_util::ceil(indices.len(), 8);
    let mut null_buf = MutableBuffer::new(num_bytes).with_bitset(num_bytes, true);
    let null_slice = null_buf.as_slice_mut();

    for i in 0..indices.len() {
        let index = indices
            .value(i)
            .to_usize()
            .ok_or_else(|| ArrowError::ComputeError("Cast to usize failed".to_string()))?;
        if !indices.is_valid(i) || values.is_null(index) {
            bit_util::unset_bit(null_slice, i);
        }
    }

    let list_data = ArrayDataBuilder::new(values.data_type().clone())
        .len(indices.len())
        .null_bit_buffer(Some(null_buf.into()))
        .offset(0)
        .add_child_data(taken.into_data());

    let list_data = unsafe { list_data.build_unchecked() };

    Ok(FixedSizeListArray::from(list_data))
}

/// The take kernel implementation for `FixedSizeBinaryArray`.
///
/// The computation is done in two steps:
/// - Compute the values buffer
/// - Compute the null buffer
fn take_fixed_size_binary<IndexType: ArrowPrimitiveType>(
    values: &FixedSizeBinaryArray,
    indices: &PrimitiveArray<IndexType>,
    size: i32,
) -> Result<FixedSizeBinaryArray, ArrowError> {
    let size_usize = usize::try_from(size).map_err(|_| {
        ArrowError::InvalidArgumentError(format!("Cannot convert size '{}' to usize", size))
    })?;

    let result_buffer = match size_usize {
        1 => take_fixed_size::<IndexType, 1>(values.values(), indices),
        2 => take_fixed_size::<IndexType, 2>(values.values(), indices),
        4 => take_fixed_size::<IndexType, 4>(values.values(), indices),
        8 => take_fixed_size::<IndexType, 8>(values.values(), indices),
        16 => take_fixed_size::<IndexType, 16>(values.values(), indices),
        _ => take_fixed_size_binary_buffer_dynamic_length(values, indices, size_usize),
    };

    let value_nulls = take_nulls(values.nulls(), indices);
    let final_nulls = NullBuffer::union(value_nulls.as_ref(), indices.nulls());
    let array_data = ArrayDataBuilder::new(DataType::FixedSizeBinary(size))
        .len(indices.len())
        .nulls(final_nulls)
        .offset(0)
        .add_buffer(result_buffer)
        .build()?;

    return Ok(FixedSizeBinaryArray::from(array_data));

    /// Implementation of the take kernel for fixed size binary arrays.
    #[inline(never)]
    fn take_fixed_size_binary_buffer_dynamic_length<IndexType: ArrowPrimitiveType>(
        values: &FixedSizeBinaryArray,
        indices: &PrimitiveArray<IndexType>,
        size_usize: usize,
    ) -> Buffer {
        let values_buffer = values.values().as_slice();
        let mut values_buffer_builder = BufferBuilder::new(indices.len() * size_usize);

        if indices.null_count() == 0 {
            let array_iter = indices.values().iter().map(|idx| {
                let offset = idx.as_usize() * size_usize;
                &values_buffer[offset..offset + size_usize]
            });
            for slice in array_iter {
                values_buffer_builder.append_slice(slice);
            }
        } else {
            // The indices nullability cannot be ignored here because the values buffer may contain
            // nulls which should not cause a panic.
            let array_iter = indices.iter().map(|idx| {
                idx.map(|idx| {
                    let offset = idx.as_usize() * size_usize;
                    &values_buffer[offset..offset + size_usize]
                })
            });
            for slice in array_iter {
                match slice {
                    None => values_buffer_builder.append_n(size_usize, 0),
                    Some(slice) => values_buffer_builder.append_slice(slice),
                }
            }
        }

        values_buffer_builder.finish()
    }
}

/// Implements the take kernel semantics over a flat [`Buffer`], interpreting it as a slice of
/// `&[[u8; N]]`, where `N` is a compile-time constant. The usage of a flat [`Buffer`] allows using
/// this kernel without an available [`ArrowPrimitiveType`] (e.g., for `[u8; 5]`).
///
/// # Using This Function in the Primitive Take Kernel
///
/// This function is basically the same as [`take_native`] but just on a flat [`Buffer`] instead of
/// the primitive [`ScalarBuffer`]. Ideally, the [`take_primitive`] kernel should just use this
/// more general function. However, the "idiomatic code" requires the
/// [feature(generic_const_exprs)](https://github.com/rust-lang/rust/issues/76560) for calling
/// `take_fixed_size<I, { size_of::<T::Native> () } >(...)`. Once this feature has been stabilized,
/// we can use this function also in the primitive kernels.
fn take_fixed_size<IndexType: ArrowPrimitiveType, const N: usize>(
    buffer: &Buffer,
    indices: &PrimitiveArray<IndexType>,
) -> Buffer {
    assert_eq!(
        buffer.len() % N,
        0,
        "Invalid array length in take_fixed_size"
    );

    let ptr = buffer.as_ptr();
    let chunk_ptr = ptr.cast::<[u8; N]>();
    let chunk_len = buffer.len() / N;
    let buffer: &[[u8; N]] = unsafe {
        // SAFETY: interpret an already valid slice as a slice of N-byte chunks. N divides buffer
        // length without remainder.
        std::slice::from_raw_parts(chunk_ptr, chunk_len)
    };

    let result_buffer = match indices.nulls().filter(|n| n.null_count() > 0) {
        Some(n) => indices
            .values()
            .iter()
            .enumerate()
            .map(|(idx, index)| match buffer.get(index.as_usize()) {
                Some(v) => *v,
                // SAFETY: idx<indices.len()
                None => match unsafe { n.inner().value_unchecked(idx) } {
                    false => [0u8; N],
                    true => panic!("Out-of-bounds index {index:?}"),
                },
            })
            .collect::<Vec<_>>(),
        None => indices
            .values()
            .iter()
            .map(|index| buffer[index.as_usize()])
            .collect::<Vec<_>>(),
    };

    let mut vec = ManuallyDrop::new(result_buffer); // Prevent de-allocation
    let ptr = vec.as_mut_ptr();
    let len = vec.len();
    let cap = vec.capacity();
    let result_buffer = unsafe {
        // SAFETY: flattening an already valid Vec.
        Vec::from_raw_parts(ptr.cast::<u8>(), len * N, cap * N)
    };

    Buffer::from_vec(result_buffer)
}

/// `take` implementation for dictionary arrays
///
/// applies `take` to the keys of the dictionary array and returns a new dictionary array
/// with the same dictionary values and reordered keys
fn take_dict<T: ArrowDictionaryKeyType, I: ArrowPrimitiveType>(
    values: &DictionaryArray<T>,
    indices: &PrimitiveArray<I>,
) -> Result<DictionaryArray<T>, ArrowError> {
    let new_keys = take_primitive(values.keys(), indices)?;
    Ok(unsafe { DictionaryArray::new_unchecked(new_keys, values.values().clone()) })
}

/// `take` implementation for run arrays
///
/// Finds physical indices for the given logical indices and builds output run array
/// by taking values in the input run_array.values at the physical indices.
/// The output run array will be run encoded on the physical indices and not on output values.
/// For e.g. an input `RunArray{ run_ends = [2,4,6,8], values=[1,2,1,2] }` and `logical_indices=[2,3,6,7]`
/// would be converted to `physical_indices=[1,1,3,3]` which will be used to build
/// output `RunArray{ run_ends=[2,4], values=[2,2] }`.
fn take_run<T: RunEndIndexType, I: ArrowPrimitiveType>(
    run_array: &RunArray<T>,
    logical_indices: &PrimitiveArray<I>,
) -> Result<RunArray<T>, ArrowError> {
    // get physical indices for the input logical indices
    let physical_indices = run_array.get_physical_indices(logical_indices.values())?;

    // Run encode the physical indices into new_run_ends_builder
    // Keep track of the physical indices to take in take_value_indices
    // `unwrap` is used in this function because the unwrapped values are bounded by the corresponding `::Native`.
    let mut new_run_ends_builder = BufferBuilder::<T::Native>::new(1);
    let mut take_value_indices = BufferBuilder::<I::Native>::new(1);
    let mut new_physical_len = 1;
    for ix in 1..physical_indices.len() {
        if physical_indices[ix] != physical_indices[ix - 1] {
            take_value_indices.append(I::Native::from_usize(physical_indices[ix - 1]).unwrap());
            new_run_ends_builder.append(T::Native::from_usize(ix).unwrap());
            new_physical_len += 1;
        }
    }
    take_value_indices
        .append(I::Native::from_usize(physical_indices[physical_indices.len() - 1]).unwrap());
    new_run_ends_builder.append(T::Native::from_usize(physical_indices.len()).unwrap());
    let new_run_ends = unsafe {
        // Safety:
        // The function builds a valid run_ends array and hence need not be validated.
        ArrayDataBuilder::new(T::DATA_TYPE)
            .len(new_physical_len)
            .null_count(0)
            .add_buffer(new_run_ends_builder.finish())
            .build_unchecked()
    };

    let take_value_indices: PrimitiveArray<I> = unsafe {
        // Safety:
        // The function builds a valid take_value_indices array and hence need not be validated.
        ArrayDataBuilder::new(I::DATA_TYPE)
            .len(new_physical_len)
            .null_count(0)
            .add_buffer(take_value_indices.finish())
            .build_unchecked()
            .into()
    };

    let new_values = take(run_array.values(), &take_value_indices, None)?;

    let builder = ArrayDataBuilder::new(run_array.data_type().clone())
        .len(physical_indices.len())
        .add_child_data(new_run_ends)
        .add_child_data(new_values.into_data());
    let array_data = unsafe {
        // Safety:
        //  This function builds a valid run array and hence can skip validation.
        builder.build_unchecked()
    };
    Ok(array_data.into())
}

/// Takes/filters a fixed size list array's inner data using the offsets of the list array.
fn take_value_indices_from_fixed_size_list<IndexType>(
    list: &FixedSizeListArray,
    indices: &PrimitiveArray<IndexType>,
    length: <UInt32Type as ArrowPrimitiveType>::Native,
) -> Result<PrimitiveArray<UInt32Type>, ArrowError>
where
    IndexType: ArrowPrimitiveType,
{
    let mut values = UInt32Builder::with_capacity(length as usize * indices.len());

    for i in 0..indices.len() {
        if indices.is_valid(i) {
            let index = indices
                .value(i)
                .to_usize()
                .ok_or_else(|| ArrowError::ComputeError("Cast to usize failed".to_string()))?;
            let start = list.value_offset(index) as <UInt32Type as ArrowPrimitiveType>::Native;

            // Safety: Range always has known length.
            unsafe {
                values.append_trusted_len_iter(start..start + length);
            }
        } else {
            values.append_nulls(length as usize);
        }
    }

    Ok(values.finish())
}

/// To avoid generating take implementations for every index type, instead we
/// only generate for UInt32 and UInt64 and coerce inputs to these types
trait ToIndices {
    type T: ArrowPrimitiveType;

    fn to_indices(&self) -> PrimitiveArray<Self::T>;
}

macro_rules! to_indices_reinterpret {
    ($t:ty, $o:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = $o;

            fn to_indices(&self) -> PrimitiveArray<$o> {
                let cast = ScalarBuffer::new(self.values().inner().clone(), 0, self.len());
                PrimitiveArray::new(cast, self.nulls().cloned())
            }
        }
    };
}

macro_rules! to_indices_identity {
    ($t:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = $t;

            fn to_indices(&self) -> PrimitiveArray<$t> {
                self.clone()
            }
        }
    };
}

macro_rules! to_indices_widening {
    ($t:ty, $o:ty) => {
        impl ToIndices for PrimitiveArray<$t> {
            type T = UInt32Type;

            fn to_indices(&self) -> PrimitiveArray<$o> {
                let cast = self.values().iter().copied().map(|x| x as _).collect();
                PrimitiveArray::new(cast, self.nulls().cloned())
            }
        }
    };
}

to_indices_widening!(UInt8Type, UInt32Type);
to_indices_widening!(Int8Type, UInt32Type);

to_indices_widening!(UInt16Type, UInt32Type);
to_indices_widening!(Int16Type, UInt32Type);

to_indices_identity!(UInt32Type);
to_indices_reinterpret!(Int32Type, UInt32Type);

to_indices_identity!(UInt64Type);
to_indices_reinterpret!(Int64Type, UInt64Type);

/// Take rows by index from [`RecordBatch`] and returns a new [`RecordBatch`] from those indexes.
///
/// This function will call [`take`] on each array of the [`RecordBatch`] and assemble a new [`RecordBatch`].
///
/// # Example
/// ```
/// # use std::sync::Arc;
/// # use arrow_array::{StringArray, Int32Array, UInt32Array, RecordBatch};
/// # use arrow_schema::{DataType, Field, Schema};
/// # use arrow_select::take::take_record_batch;
/// let schema = Arc::new(Schema::new(vec![
///     Field::new("a", DataType::Int32, true),
///     Field::new("b", DataType::Utf8, true),
/// ]));
/// let batch = RecordBatch::try_new(
///     schema.clone(),
///     vec![
///         Arc::new(Int32Array::from_iter_values(0..20)),
///         Arc::new(StringArray::from_iter_values(
///             (0..20).map(|i| format!("str-{}", i)),
///         )),
///     ],
/// )
/// .unwrap();
///
/// let indices = UInt32Array::from(vec![1, 5, 10]);
/// let taken = take_record_batch(&batch, &indices).unwrap();
///
/// let expected = RecordBatch::try_new(
///     schema,
///     vec![
///         Arc::new(Int32Array::from(vec![1, 5, 10])),
///         Arc::new(StringArray::from(vec!["str-1", "str-5", "str-10"])),
///     ],
/// )
/// .unwrap();
/// assert_eq!(taken, expected);
/// ```
pub fn take_record_batch(
    record_batch: &RecordBatch,
    indices: &dyn Array,
) -> Result<RecordBatch, ArrowError> {
    let columns = record_batch
        .columns()
        .iter()
        .map(|c| take(c, indices, None))
        .collect::<Result<Vec<_>, _>>()?;
    RecordBatch::try_new(record_batch.schema(), columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::builder::*;
    use arrow_buffer::{IntervalDayTime, IntervalMonthDayNano};
    use arrow_data::ArrayData;
    use arrow_schema::{Field, Fields, TimeUnit, UnionFields};
    use num_traits::ToPrimitive;

    fn test_take_decimal_arrays(
        data: Vec<Option<i128>>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<i128>>,
        precision: &u8,
        scale: &i8,
    ) -> Result<(), ArrowError> {
        let output = data
            .into_iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(*precision, *scale)
            .unwrap();

        let expected = expected_data
            .into_iter()
            .collect::<Decimal128Array>()
            .with_precision_and_scale(*precision, *scale)
            .unwrap();

        let expected = Arc::new(expected) as ArrayRef;
        let output = take(&output, index, options).unwrap();
        assert_eq!(&output, &expected);
        Ok(())
    }

    fn test_take_boolean_arrays(
        data: Vec<Option<bool>>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<bool>>,
    ) {
        let output = BooleanArray::from(data);
        let expected = Arc::new(BooleanArray::from(expected_data)) as ArrayRef;
        let output = take(&output, index, options).unwrap();
        assert_eq!(&output, &expected)
    }

    fn test_take_primitive_arrays<T>(
        data: Vec<Option<T::Native>>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) -> Result<(), ArrowError>
    where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = Arc::new(PrimitiveArray::<T>::from(expected_data)) as ArrayRef;
        let output = take(&output, index, options)?;
        assert_eq!(&output, &expected);
        Ok(())
    }

    fn test_take_primitive_arrays_non_null<T>(
        data: Vec<T::Native>,
        index: &UInt32Array,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) -> Result<(), ArrowError>
    where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<T::Native>>,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = Arc::new(PrimitiveArray::<T>::from(expected_data)) as ArrayRef;
        let output = take(&output, index, options)?;
        assert_eq!(&output, &expected);
        Ok(())
    }

    fn test_take_impl_primitive_arrays<T, I>(
        data: Vec<Option<T::Native>>,
        index: &PrimitiveArray<I>,
        options: Option<TakeOptions>,
        expected_data: Vec<Option<T::Native>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
        I: ArrowPrimitiveType,
    {
        let output = PrimitiveArray::<T>::from(data);
        let expected = PrimitiveArray::<T>::from(expected_data);
        let output = take(&output, index, options).unwrap();
        let output = output.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
        assert_eq!(output, &expected)
    }

    // create a simple struct for testing purposes
    fn create_test_struct(values: Vec<Option<(Option<bool>, Option<i32>)>>) -> StructArray {
        let mut struct_builder = StructBuilder::new(
            Fields::from(vec![
                Field::new("a", DataType::Boolean, true),
                Field::new("b", DataType::Int32, true),
            ]),
            vec![
                Box::new(BooleanBuilder::with_capacity(values.len())),
                Box::new(Int32Builder::with_capacity(values.len())),
            ],
        );

        for value in values {
            struct_builder
                .field_builder::<BooleanBuilder>(0)
                .unwrap()
                .append_option(value.and_then(|v| v.0));
            struct_builder
                .field_builder::<Int32Builder>(1)
                .unwrap()
                .append_option(value.and_then(|v| v.1));
            struct_builder.append(value.is_some());
        }
        struct_builder.finish()
    }

    #[test]
    fn test_take_decimal128_non_null_indices() {
        let index = UInt32Array::from(vec![0, 5, 3, 1, 4, 2]);
        let precision: u8 = 10;
        let scale: i8 = 5;
        test_take_decimal_arrays(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            &index,
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
            &precision,
            &scale,
        )
        .unwrap();
    }

    #[test]
    fn test_take_decimal128() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);
        let precision: u8 = 10;
        let scale: i8 = 5;
        test_take_decimal_arrays(
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)],
            &index,
            None,
            vec![Some(3), None, Some(1), Some(3), Some(2)],
            &precision,
            &scale,
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive_non_null_indices() {
        let index = UInt32Array::from(vec![0, 5, 3, 1, 4, 2]);
        test_take_primitive_arrays::<Int8Type>(
            vec![None, Some(3), Some(5), Some(2), Some(3), None],
            &index,
            None,
            vec![None, None, Some(2), Some(3), Some(3), Some(5)],
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive_non_null_values() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);
        test_take_primitive_arrays::<Int8Type>(
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)],
            &index,
            None,
            vec![Some(3), None, Some(1), Some(3), Some(2)],
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive_non_null() {
        let index = UInt32Array::from(vec![0, 5, 3, 1, 4, 2]);
        test_take_primitive_arrays::<Int8Type>(
            vec![Some(0), Some(3), Some(5), Some(2), Some(3), Some(1)],
            &index,
            None,
            vec![Some(0), Some(1), Some(2), Some(3), Some(3), Some(5)],
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive_nullable_indices_non_null_values_with_offset() {
        let index = UInt32Array::from(vec![Some(0), Some(1), Some(2), Some(3), None, None]);
        let index = index.slice(2, 4);
        let index = index.as_any().downcast_ref::<UInt32Array>().unwrap();

        assert_eq!(
            index,
            &UInt32Array::from(vec![Some(2), Some(3), None, None])
        );

        test_take_primitive_arrays_non_null::<Int64Type>(
            vec![0, 10, 20, 30, 40, 50],
            index,
            None,
            vec![Some(20), Some(30), None, None],
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive_nullable_indices_nullable_values_with_offset() {
        let index = UInt32Array::from(vec![Some(0), Some(1), Some(2), Some(3), None, None]);
        let index = index.slice(2, 4);
        let index = index.as_any().downcast_ref::<UInt32Array>().unwrap();

        assert_eq!(
            index,
            &UInt32Array::from(vec![Some(2), Some(3), None, None])
        );

        test_take_primitive_arrays::<Int64Type>(
            vec![None, None, Some(20), Some(30), Some(40), Some(50)],
            index,
            None,
            vec![Some(20), Some(30), None, None],
        )
        .unwrap();
    }

    #[test]
    fn test_take_primitive() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int8
        test_take_primitive_arrays::<Int8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // int16
        test_take_primitive_arrays::<Int16Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // int32
        test_take_primitive_arrays::<Int32Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // uint8
        test_take_primitive_arrays::<UInt8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // uint16
        test_take_primitive_arrays::<UInt16Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // uint32
        test_take_primitive_arrays::<UInt32Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        )
        .unwrap();

        // int64
        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // interval_year_month
        test_take_primitive_arrays::<IntervalYearMonthType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // interval_day_time
        let v1 = IntervalDayTime::new(0, 0);
        let v2 = IntervalDayTime::new(2, 0);
        let v3 = IntervalDayTime::new(-15, 0);
        test_take_primitive_arrays::<IntervalDayTimeType>(
            vec![Some(v1), None, Some(v2), Some(v3), None],
            &index,
            None,
            vec![Some(v3), None, None, Some(v3), Some(v2)],
        )
        .unwrap();

        // interval_month_day_nano
        let v1 = IntervalMonthDayNano::new(0, 0, 0);
        let v2 = IntervalMonthDayNano::new(2, 0, 0);
        let v3 = IntervalMonthDayNano::new(-15, 0, 0);
        test_take_primitive_arrays::<IntervalMonthDayNanoType>(
            vec![Some(v1), None, Some(v2), Some(v3), None],
            &index,
            None,
            vec![Some(v3), None, None, Some(v3), Some(v2)],
        )
        .unwrap();

        // duration_second
        test_take_primitive_arrays::<DurationSecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // duration_millisecond
        test_take_primitive_arrays::<DurationMillisecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // duration_microsecond
        test_take_primitive_arrays::<DurationMicrosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // duration_nanosecond
        test_take_primitive_arrays::<DurationNanosecondType>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        )
        .unwrap();

        // float32
        test_take_primitive_arrays::<Float32Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        )
        .unwrap();

        // float64
        test_take_primitive_arrays::<Float64Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        )
        .unwrap();
    }

    #[test]
    fn test_take_preserve_timezone() {
        let index = Int64Array::from(vec![Some(0), None]);

        let input = TimestampNanosecondArray::from(vec![
            1_639_715_368_000_000_000,
            1_639_715_368_000_000_000,
        ])
        .with_timezone("UTC".to_string());
        let result = take(&input, &index, None).unwrap();
        match result.data_type() {
            DataType::Timestamp(TimeUnit::Nanosecond, tz) => {
                assert_eq!(tz.clone(), Some("UTC".into()))
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_take_impl_primitive_with_int64_indices() {
        let index = Int64Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int16
        test_take_impl_primitive_arrays::<Int16Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // int64
        test_take_impl_primitive_arrays::<Int64Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // uint64
        test_take_impl_primitive_arrays::<UInt64Type, Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // duration_millisecond
        test_take_impl_primitive_arrays::<DurationMillisecondType, Int64Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_impl_primitive_arrays::<Float32Type, Int64Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );
    }

    #[test]
    fn test_take_impl_primitive_with_uint8_indices() {
        let index = UInt8Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);

        // int16
        test_take_impl_primitive_arrays::<Int16Type, UInt8Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            None,
            vec![Some(3), None, None, Some(3), Some(2)],
        );

        // duration_millisecond
        test_take_impl_primitive_arrays::<DurationMillisecondType, UInt8Type>(
            vec![Some(0), None, Some(2), Some(-15), None],
            &index,
            None,
            vec![Some(-15), None, None, Some(-15), Some(2)],
        );

        // float32
        test_take_impl_primitive_arrays::<Float32Type, UInt8Type>(
            vec![Some(0.0), None, Some(2.21), Some(-3.1), None],
            &index,
            None,
            vec![Some(-3.1), None, None, Some(-3.1), Some(2.21)],
        );
    }

    #[test]
    fn test_take_bool() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2)]);
        // boolean
        test_take_boolean_arrays(
            vec![Some(false), None, Some(true), Some(false), None],
            &index,
            None,
            vec![Some(false), None, None, Some(false), Some(true)],
        );
    }

    #[test]
    fn test_take_bool_nullable_index() {
        // indices where the masked invalid elements would be out of bounds
        let index_data = ArrayData::try_new(
            DataType::UInt32,
            6,
            Some(Buffer::from_iter(vec![
                false, true, false, true, false, true,
            ])),
            0,
            vec![Buffer::from_iter(vec![99, 0, 999, 1, 9999, 2])],
            vec![],
        )
        .unwrap();
        let index = UInt32Array::from(index_data);
        test_take_boolean_arrays(
            vec![Some(true), None, Some(false)],
            &index,
            None,
            vec![None, Some(true), None, None, None, Some(false)],
        );
    }

    #[test]
    fn test_take_bool_nullable_index_nonnull_values() {
        // indices where the masked invalid elements would be out of bounds
        let index_data = ArrayData::try_new(
            DataType::UInt32,
            6,
            Some(Buffer::from_iter(vec![
                false, true, false, true, false, true,
            ])),
            0,
            vec![Buffer::from_iter(vec![99, 0, 999, 1, 9999, 2])],
            vec![],
        )
        .unwrap();
        let index = UInt32Array::from(index_data);
        test_take_boolean_arrays(
            vec![Some(true), Some(true), Some(false)],
            &index,
            None,
            vec![None, Some(true), None, Some(true), None, Some(false)],
        );
    }

    #[test]
    fn test_take_bool_with_offset() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(2), None]);
        let index = index.slice(2, 4);
        let index = index
            .as_any()
            .downcast_ref::<PrimitiveArray<UInt32Type>>()
            .unwrap();

        // boolean
        test_take_boolean_arrays(
            vec![Some(false), None, Some(true), Some(false), None],
            index,
            None,
            vec![None, Some(false), Some(true), None],
        );
    }

    fn _test_take_string<'a, K>()
    where
        K: Array + PartialEq + From<Vec<Option<&'a str>>> + 'static,
    {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(4)]);

        let array = K::from(vec![
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            Some("five"),
        ]);
        let actual = take(&array, &index, None).unwrap();
        assert_eq!(actual.len(), index.len());

        let actual = actual.as_any().downcast_ref::<K>().unwrap();

        let expected = K::from(vec![Some("four"), None, None, Some("four"), Some("five")]);

        assert_eq!(actual, &expected);
    }

    #[test]
    fn test_take_string() {
        _test_take_string::<StringArray>()
    }

    #[test]
    fn test_take_large_string() {
        _test_take_string::<LargeStringArray>()
    }

    #[test]
    fn test_take_slice_string() {
        let strings = StringArray::from(vec![Some("hello"), None, Some("world"), None, Some("hi")]);
        let indices = Int32Array::from(vec![Some(0), Some(1), None, Some(0), Some(2)]);
        let indices_slice = indices.slice(1, 4);
        let expected = StringArray::from(vec![None, None, Some("hello"), Some("world")]);
        let result = take(&strings, &indices_slice, None).unwrap();
        assert_eq!(result.as_ref(), &expected);
    }

    fn _test_byte_view<T>()
    where
        T: ByteViewType,
        str: AsRef<T::Native>,
        T::Native: PartialEq,
    {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(4), Some(2)]);
        let array = {
            // ["hello", "world", null, "large payload over 12 bytes", "lulu"]
            let mut builder = GenericByteViewBuilder::<T>::new();
            builder.append_value("hello");
            builder.append_value("world");
            builder.append_null();
            builder.append_value("large payload over 12 bytes");
            builder.append_value("lulu");
            builder.finish()
        };

        let actual = take(&array, &index, None).unwrap();

        assert_eq!(actual.len(), index.len());

        let expected = {
            // ["large payload over 12 bytes", null, "world", "large payload over 12 bytes", "lulu", null]
            let mut builder = GenericByteViewBuilder::<T>::new();
            builder.append_value("large payload over 12 bytes");
            builder.append_null();
            builder.append_value("world");
            builder.append_value("large payload over 12 bytes");
            builder.append_value("lulu");
            builder.append_null();
            builder.finish()
        };

        assert_eq!(actual.as_ref(), &expected);
    }

    #[test]
    fn test_take_string_view() {
        _test_byte_view::<StringViewType>()
    }

    #[test]
    fn test_take_binary_view() {
        _test_byte_view::<BinaryViewType>()
    }

    macro_rules! test_take_list {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,0,0], [-1,-2,-1], [], [2,3]]
            let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 3]).into_data();
            // Construct offsets
            let value_offsets: [$offset_type; 5] = [0, 3, 6, 6, 8];
            let value_offsets = Buffer::from_slice_ref(&value_offsets);
            // Construct a list array from the above two
            let list_data_type =
                DataType::$list_data_type(Arc::new(Field::new_list_field(DataType::Int32, false)));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(4)
                .add_buffer(value_offsets)
                .add_child_data(value_data)
                .build()
                .unwrap();
            let list_array = $list_array_type::from(list_data);

            // index returns: [[2,3], null, [-1,-2,-1], [], [0,0,0]]
            let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(2), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type = a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [[2,3], null, [-1,-2,-1], [], [0,0,0]]
            let expected_data = Int32Array::from(vec![
                Some(2),
                Some(3),
                Some(-1),
                Some(-2),
                Some(-1),
                Some(0),
                Some(0),
                Some(0),
            ])
            .into_data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 2, 2, 5, 5, 8];
            let expected_offsets = Buffer::from_slice_ref(&expected_offsets);
            // construct list array from the two
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                // null buffer remains the same as only the indices have nulls
                .nulls(index.nulls().cloned())
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build()
                .unwrap();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    macro_rules! test_take_list_with_value_nulls {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,null,0], [-1,-2,3], [null], [5,null]]
            let value_data = Int32Array::from(vec![
                Some(0),
                None,
                Some(0),
                Some(-1),
                Some(-2),
                Some(3),
                None,
                Some(5),
                None,
            ])
            .into_data();
            // Construct offsets
            let value_offsets: [$offset_type; 5] = [0, 3, 6, 7, 9];
            let value_offsets = Buffer::from_slice_ref(&value_offsets);
            // Construct a list array from the above two
            let list_data_type =
                DataType::$list_data_type(Arc::new(Field::new_list_field(DataType::Int32, true)));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(4)
                .add_buffer(value_offsets)
                .null_bit_buffer(Some(Buffer::from([0b11111111])))
                .add_child_data(value_data)
                .build()
                .unwrap();
            let list_array = $list_array_type::from(list_data);

            // index returns: [[null], null, [-1,-2,3], [2,null], [0,null,0]]
            let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type = a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [[null], null, [-1,-2,3], [5,null], [0,null,0]]
            let expected_data = Int32Array::from(vec![
                None,
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
                Some(0),
                None,
                Some(0),
            ])
            .into_data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 1, 1, 4, 6, 9];
            let expected_offsets = Buffer::from_slice_ref(&expected_offsets);
            // construct list array from the two
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                // null buffer remains the same as only the indices have nulls
                .nulls(index.nulls().cloned())
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build()
                .unwrap();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    macro_rules! test_take_list_with_nulls {
        ($offset_type:ty, $list_data_type:ident, $list_array_type:ident) => {{
            // Construct a value array, [[0,null,0], [-1,-2,3], null, [5,null]]
            let value_data = Int32Array::from(vec![
                Some(0),
                None,
                Some(0),
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
            ])
            .into_data();
            // Construct offsets
            let value_offsets: [$offset_type; 5] = [0, 3, 6, 6, 8];
            let value_offsets = Buffer::from_slice_ref(&value_offsets);
            // Construct a list array from the above two
            let list_data_type =
                DataType::$list_data_type(Arc::new(Field::new_list_field(DataType::Int32, true)));
            let list_data = ArrayData::builder(list_data_type.clone())
                .len(4)
                .add_buffer(value_offsets)
                .null_bit_buffer(Some(Buffer::from([0b11111011])))
                .add_child_data(value_data)
                .build()
                .unwrap();
            let list_array = $list_array_type::from(list_data);

            // index returns: [null, null, [-1,-2,3], [5,null], [0,null,0]]
            let index = UInt32Array::from(vec![Some(2), None, Some(1), Some(3), Some(0)]);

            let a = take(&list_array, &index, None).unwrap();
            let a: &$list_array_type = a.as_any().downcast_ref::<$list_array_type>().unwrap();

            // construct a value array with expected results:
            // [null, null, [-1,-2,3], [5,null], [0,null,0]]
            let expected_data = Int32Array::from(vec![
                Some(-1),
                Some(-2),
                Some(3),
                Some(5),
                None,
                Some(0),
                None,
                Some(0),
            ])
            .into_data();
            // construct offsets
            let expected_offsets: [$offset_type; 6] = [0, 0, 0, 3, 5, 8];
            let expected_offsets = Buffer::from_slice_ref(&expected_offsets);
            // construct list array from the two
            let mut null_bits: [u8; 1] = [0; 1];
            bit_util::set_bit(&mut null_bits, 2);
            bit_util::set_bit(&mut null_bits, 3);
            bit_util::set_bit(&mut null_bits, 4);
            let expected_list_data = ArrayData::builder(list_data_type)
                .len(5)
                // null buffer must be recalculated as both values and indices have nulls
                .null_bit_buffer(Some(Buffer::from(null_bits)))
                .add_buffer(expected_offsets)
                .add_child_data(expected_data)
                .build()
                .unwrap();
            let expected_list_array = $list_array_type::from(expected_list_data);

            assert_eq!(a, &expected_list_array);
        }};
    }

    fn test_take_list_view_generic<OffsetType: OffsetSizeTrait, ValuesType: ArrowPrimitiveType, F>(
        values: Vec<Option<Vec<Option<ValuesType::Native>>>>,
        take_indices: Vec<Option<usize>>,
        expected: Vec<Option<Vec<Option<ValuesType::Native>>>>,
        mapper: F,
    ) where
        F: Fn(GenericListViewArray<OffsetType>) -> GenericListViewArray<OffsetType>,
    {
        let mut list_view_array =
            GenericListViewBuilder::<OffsetType, _>::new(PrimitiveBuilder::<ValuesType>::new());

        for value in values {
            list_view_array.append_option(value);
        }
        let list_view_array = list_view_array.finish();
        let list_view_array = mapper(list_view_array);

        let mut indices = UInt64Builder::new();
        for idx in take_indices {
            indices.append_option(idx.map(|i| i.to_u64().unwrap()));
        }
        let indices = indices.finish();

        let taken = take(&list_view_array, &indices, None)
            .unwrap()
            .as_list_view()
            .clone();

        let mut expected_array =
            GenericListViewBuilder::<OffsetType, _>::new(PrimitiveBuilder::<ValuesType>::new());
        for value in expected {
            expected_array.append_option(value);
        }
        let expected_array = expected_array.finish();

        assert_eq!(taken, expected_array);
    }

    macro_rules! list_view_test_case {
        (values: $values:expr, indices: $indices:expr, expected: $expected: expr) => {{
            test_take_list_view_generic::<i32, Int8Type, _>($values, $indices, $expected, |x| x);
            test_take_list_view_generic::<i64, Int8Type, _>($values, $indices, $expected, |x| x);
        }};
        (values: $values:expr, transform: $fn:expr, indices: $indices:expr, expected: $expected: expr) => {{
            test_take_list_view_generic::<i32, Int8Type, _>($values, $indices, $expected, $fn);
            test_take_list_view_generic::<i64, Int8Type, _>($values, $indices, $expected, $fn);
        }};
    }

    fn do_take_fixed_size_list_test<T>(
        length: <Int32Type as ArrowPrimitiveType>::Native,
        input_data: Vec<Option<Vec<Option<T::Native>>>>,
        indices: Vec<<UInt32Type as ArrowPrimitiveType>::Native>,
        expected_data: Vec<Option<Vec<Option<T::Native>>>>,
    ) where
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        let indices = UInt32Array::from(indices);

        let input_array = FixedSizeListArray::from_iter_primitive::<T, _, _>(input_data, length);

        let output = take_fixed_size_list(&input_array, &indices, length as u32).unwrap();

        let expected = FixedSizeListArray::from_iter_primitive::<T, _, _>(expected_data, length);

        assert_eq!(&output, &expected)
    }

    #[test]
    fn test_take_list() {
        test_take_list!(i32, List, ListArray);
    }

    #[test]
    fn test_take_large_list() {
        test_take_list!(i64, LargeList, LargeListArray);
    }

    #[test]
    fn test_take_list_with_value_nulls() {
        test_take_list_with_value_nulls!(i32, List, ListArray);
    }

    #[test]
    fn test_take_large_list_with_value_nulls() {
        test_take_list_with_value_nulls!(i64, LargeList, LargeListArray);
    }

    #[test]
    fn test_test_take_list_with_nulls() {
        test_take_list_with_nulls!(i32, List, ListArray);
    }

    #[test]
    fn test_test_take_large_list_with_nulls() {
        test_take_list_with_nulls!(i64, LargeList, LargeListArray);
    }

    #[test]
    fn test_test_take_list_view_reversed() {
        // Take reversed indices
        list_view_test_case! {
            values: vec![
                Some(vec![Some(1), None, Some(3)]),
                None,
                Some(vec![Some(7), Some(8), None]),
            ],
            indices: vec![Some(2), Some(1), Some(0)],
            expected: vec![
                Some(vec![Some(7), Some(8), None]),
                None,
                Some(vec![Some(1), None, Some(3)]),
            ]
        }
    }

    #[test]
    fn test_take_list_view_null_indices() {
        // Take with null indices
        list_view_test_case! {
            values: vec![
                Some(vec![Some(1), None, Some(3)]),
                None,
                Some(vec![Some(7), Some(8), None]),
            ],
            indices: vec![None, Some(0), None],
            expected: vec![None, Some(vec![Some(1), None, Some(3)]), None]
        }
    }

    #[test]
    fn test_take_list_view_null_values() {
        // Take at null values
        list_view_test_case! {
            values: vec![
                Some(vec![Some(1), None, Some(3)]),
                None,
                Some(vec![Some(7), Some(8), None]),
            ],
            indices: vec![Some(1), Some(1), Some(1), None, None],
            expected: vec![None; 5]
        }
    }

    #[test]
    fn test_take_list_view_sliced() {
        // Take null indices/values, with slicing.
        list_view_test_case! {
            values: vec![
                Some(vec![Some(1)]),
                None,
                None,
                Some(vec![Some(2), Some(3)]),
                Some(vec![Some(4), Some(5)]),
                None,
            ],
            transform: |l| l.slice(2, 4),
            indices: vec![Some(0), Some(3), None, Some(1), Some(2)],
            expected: vec![
                None, None, None, Some(vec![Some(2), Some(3)]), Some(vec![Some(4), Some(5)])
            ]
        }
    }

    #[test]
    fn test_take_fixed_size_list() {
        do_take_fixed_size_list_test::<Int32Type>(
            3,
            vec![
                Some(vec![None, Some(1), Some(2)]),
                Some(vec![Some(3), Some(4), None]),
                Some(vec![Some(6), Some(7), Some(8)]),
            ],
            vec![2, 1, 0],
            vec![
                Some(vec![Some(6), Some(7), Some(8)]),
                Some(vec![Some(3), Some(4), None]),
                Some(vec![None, Some(1), Some(2)]),
            ],
        );

        do_take_fixed_size_list_test::<UInt8Type>(
            1,
            vec![
                Some(vec![Some(1)]),
                Some(vec![Some(2)]),
                Some(vec![Some(3)]),
                Some(vec![Some(4)]),
                Some(vec![Some(5)]),
                Some(vec![Some(6)]),
                Some(vec![Some(7)]),
                Some(vec![Some(8)]),
            ],
            vec![2, 7, 0],
            vec![
                Some(vec![Some(3)]),
                Some(vec![Some(8)]),
                Some(vec![Some(1)]),
            ],
        );

        do_take_fixed_size_list_test::<UInt64Type>(
            3,
            vec![
                Some(vec![Some(10), Some(11), Some(12)]),
                Some(vec![Some(13), Some(14), Some(15)]),
                None,
                Some(vec![Some(16), Some(17), Some(18)]),
            ],
            vec![3, 2, 1, 2, 0],
            vec![
                Some(vec![Some(16), Some(17), Some(18)]),
                None,
                Some(vec![Some(13), Some(14), Some(15)]),
                None,
                Some(vec![Some(10), Some(11), Some(12)]),
            ],
        );
    }

    #[test]
    fn test_take_fixed_size_binary_with_nulls_indices() {
        let fsb = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [
                Some(vec![0x01, 0x01, 0x01, 0x01]),
                Some(vec![0x02, 0x02, 0x02, 0x02]),
                Some(vec![0x03, 0x03, 0x03, 0x03]),
                Some(vec![0x04, 0x04, 0x04, 0x04]),
            ]
            .into_iter(),
            4,
        )
        .unwrap();

        // The two middle indices are null -> Should be null in the output.
        let indices = UInt32Array::from(vec![Some(0), None, None, Some(3)]);

        let result = take_fixed_size_binary(&fsb, &indices, 4).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.null_count(), 2);
        assert_eq!(
            result.nulls().unwrap().iter().collect::<Vec<_>>(),
            vec![true, false, false, true]
        );
    }

    /// The [`take_fixed_size_binary`] kernel contains optimizations that provide a faster
    /// implementation for commonly-used value lengths. This test uses a value length that is not
    /// optimized to test both code paths.
    #[test]
    fn test_take_fixed_size_binary_with_nulls_indices_not_optimized_length() {
        let fsb = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
            [
                Some(vec![0x01, 0x01, 0x01, 0x01, 0x01]),
                Some(vec![0x02, 0x02, 0x02, 0x02, 0x01]),
                Some(vec![0x03, 0x03, 0x03, 0x03, 0x01]),
                Some(vec![0x04, 0x04, 0x04, 0x04, 0x01]),
            ]
            .into_iter(),
            5,
        )
        .unwrap();

        // The two middle indices are null -> Should be null in the output.
        let indices = UInt32Array::from(vec![Some(0), None, None, Some(3)]);

        let result = take_fixed_size_binary(&fsb, &indices, 5).unwrap();
        assert_eq!(result.len(), 4);
        assert_eq!(result.null_count(), 2);
        assert_eq!(
            result.nulls().unwrap().iter().collect::<Vec<_>>(),
            vec![true, false, false, true]
        );
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 1000")]
    fn test_take_list_out_of_bounds() {
        // Construct a value array, [[0,0,0], [-1,-2,-1], [2,3]]
        let value_data = Int32Array::from(vec![0, 0, 0, -1, -2, -1, 2, 3]).into_data();
        // Construct offsets
        let value_offsets = Buffer::from_slice_ref([0, 3, 6, 8]);
        // Construct a list array from the above two
        let list_data_type =
            DataType::List(Arc::new(Field::new_list_field(DataType::Int32, false)));
        let list_data = ArrayData::builder(list_data_type)
            .len(3)
            .add_buffer(value_offsets)
            .add_child_data(value_data)
            .build()
            .unwrap();
        let list_array = ListArray::from(list_data);

        let index = UInt32Array::from(vec![1000]);

        // A panic is expected here since we have not supplied the check_bounds
        // option.
        take(&list_array, &index, None).unwrap();
    }

    #[test]
    fn test_take_map() {
        let values = Int32Array::from(vec![1, 2, 3, 4]);
        let array =
            MapArray::new_from_strings(vec!["a", "b", "c", "a"].into_iter(), &values, &[0, 3, 4])
                .unwrap();

        let index = UInt32Array::from(vec![0]);

        let result = take(&array, &index, None).unwrap();
        let expected: ArrayRef = Arc::new(
            MapArray::new_from_strings(
                vec!["a", "b", "c"].into_iter(),
                &values.slice(0, 3),
                &[0, 3],
            )
            .unwrap(),
        );
        assert_eq!(&expected, &result);
    }

    #[test]
    fn test_take_struct() {
        let array = create_test_struct(vec![
            Some((Some(true), Some(42))),
            Some((Some(false), Some(28))),
            Some((Some(false), Some(19))),
            Some((Some(true), Some(31))),
            None,
        ]);

        let index = UInt32Array::from(vec![0, 3, 1, 0, 2, 4]);
        let actual = take(&array, &index, None).unwrap();
        let actual: &StructArray = actual.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), actual.len());
        assert_eq!(1, actual.null_count());

        let expected = create_test_struct(vec![
            Some((Some(true), Some(42))),
            Some((Some(true), Some(31))),
            Some((Some(false), Some(28))),
            Some((Some(true), Some(42))),
            Some((Some(false), Some(19))),
            None,
        ]);

        assert_eq!(&expected, actual);

        let nulls = NullBuffer::from(&[false, true, false, true, false, true]);
        let empty_struct_arr = StructArray::new_empty_fields(6, Some(nulls));
        let index = UInt32Array::from(vec![0, 2, 1, 4]);
        let actual = take(&empty_struct_arr, &index, None).unwrap();

        let expected_nulls = NullBuffer::from(&[false, false, true, false]);
        let expected_struct_arr = StructArray::new_empty_fields(4, Some(expected_nulls));
        assert_eq!(&expected_struct_arr, actual.as_struct());
    }

    #[test]
    fn test_take_struct_with_null_indices() {
        let array = create_test_struct(vec![
            Some((Some(true), Some(42))),
            Some((Some(false), Some(28))),
            Some((Some(false), Some(19))),
            Some((Some(true), Some(31))),
            None,
        ]);

        let index = UInt32Array::from(vec![None, Some(3), Some(1), None, Some(0), Some(4)]);
        let actual = take(&array, &index, None).unwrap();
        let actual: &StructArray = actual.as_any().downcast_ref::<StructArray>().unwrap();
        assert_eq!(index.len(), actual.len());
        assert_eq!(3, actual.null_count()); // 2 because of indices, 1 because of struct array

        let expected = create_test_struct(vec![
            None,
            Some((Some(true), Some(31))),
            Some((Some(false), Some(28))),
            None,
            Some((Some(true), Some(42))),
            None,
        ]);

        assert_eq!(&expected, actual);
    }

    #[test]
    fn test_take_out_of_bounds() {
        let index = UInt32Array::from(vec![Some(3), None, Some(1), Some(3), Some(6)]);
        let take_opt = TakeOptions { check_bounds: true };

        // int64
        let result = test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), None, Some(2), Some(3), None],
            &index,
            Some(take_opt),
            vec![None],
        );
        assert!(result.is_err());
    }

    #[test]
    #[should_panic(expected = "index out of bounds: the len is 4 but the index is 1000")]
    fn test_take_out_of_bounds_panic() {
        let index = UInt32Array::from(vec![Some(1000)]);

        test_take_primitive_arrays::<Int64Type>(
            vec![Some(0), Some(1), Some(2), Some(3)],
            &index,
            None,
            vec![None],
        )
        .unwrap();
    }

    #[test]
    fn test_null_array_smaller_than_indices() {
        let values = NullArray::new(2);
        let indices = UInt32Array::from(vec![Some(0), None, Some(15)]);

        let result = take(&values, &indices, None).unwrap();
        let expected: ArrayRef = Arc::new(NullArray::new(3));
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_null_array_larger_than_indices() {
        let values = NullArray::new(5);
        let indices = UInt32Array::from(vec![Some(0), None, Some(15)]);

        let result = take(&values, &indices, None).unwrap();
        let expected: ArrayRef = Arc::new(NullArray::new(3));
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_null_array_indices_out_of_bounds() {
        let values = NullArray::new(5);
        let indices = UInt32Array::from(vec![Some(0), None, Some(15)]);

        let result = take(&values, &indices, Some(TakeOptions { check_bounds: true }));
        assert_eq!(
            result.unwrap_err().to_string(),
            "Compute error: Array index out of bounds, cannot get item at index 15 from 5 entries"
        );
    }

    #[test]
    fn test_take_dict() {
        let mut dict_builder = StringDictionaryBuilder::<Int16Type>::new();

        dict_builder.append("foo").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("").unwrap();
        dict_builder.append_null();
        dict_builder.append("foo").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("bar").unwrap();
        dict_builder.append("foo").unwrap();

        let array = dict_builder.finish();
        let dict_values = array.values().clone();
        let dict_values = dict_values.as_any().downcast_ref::<StringArray>().unwrap();

        let indices = UInt32Array::from(vec![
            Some(0), // first "foo"
            Some(7), // last "foo"
            None,    // null index should return null
            Some(5), // second "bar"
            Some(6), // another "bar"
            Some(2), // empty string
            Some(3), // input is null at this index
        ]);

        let result = take(&array, &indices, None).unwrap();
        let result = result
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .unwrap();

        let result_values: StringArray = result.values().to_data().into();

        // dictionary values should stay the same
        let expected_values = StringArray::from(vec!["foo", "bar", ""]);
        assert_eq!(&expected_values, dict_values);
        assert_eq!(&expected_values, &result_values);

        let expected_keys = Int16Array::from(vec![
            Some(0),
            Some(0),
            None,
            Some(1),
            Some(1),
            Some(2),
            None,
        ]);
        assert_eq!(result.keys(), &expected_keys);
    }

    fn build_generic_list<S, T>(data: Vec<Option<Vec<T::Native>>>) -> GenericListArray<S>
    where
        S: OffsetSizeTrait + 'static,
        T: ArrowPrimitiveType,
        PrimitiveArray<T>: From<Vec<Option<T::Native>>>,
    {
        GenericListArray::from_iter_primitive::<T, _, _>(
            data.iter()
                .map(|x| x.as_ref().map(|x| x.iter().map(|x| Some(*x)))),
        )
    }

    fn test_take_sliced_list_generic<S: OffsetSizeTrait + 'static>() {
        let list = build_generic_list::<S, Int32Type>(vec![
            Some(vec![0, 1]),
            Some(vec![2, 3, 4]),
            None,
            Some(vec![]),
            Some(vec![5, 6]),
            Some(vec![7]),
        ]);
        let sliced = list.slice(1, 4);
        let indices = UInt32Array::from(vec![Some(3), Some(0), None, Some(2), Some(1)]);

        let taken = take(&sliced, &indices, None).unwrap();
        let taken = taken.as_list::<S>();

        let expected = build_generic_list::<S, Int32Type>(vec![
            Some(vec![5, 6]),
            Some(vec![2, 3, 4]),
            None,
            Some(vec![]),
            None,
        ]);

        assert_eq!(taken, &expected);
    }

    fn test_take_sliced_list_with_value_nulls_generic<S: OffsetSizeTrait + 'static>() {
        let list = GenericListArray::<S>::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(10)]),
            Some(vec![None, Some(1)]),
            None,
            Some(vec![Some(2), None]),
            Some(vec![]),
            Some(vec![Some(3)]),
        ]);
        let sliced = list.slice(1, 4);
        let indices = UInt32Array::from(vec![Some(2), Some(0), None, Some(3), Some(1)]);

        let taken = take(&sliced, &indices, None).unwrap();
        let taken = taken.as_list::<S>();

        let expected = GenericListArray::<S>::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(2), None]),
            Some(vec![None, Some(1)]),
            None,
            Some(vec![]),
            None,
        ]);

        assert_eq!(taken, &expected);
    }

    #[test]
    fn test_take_sliced_list() {
        test_take_sliced_list_generic::<i32>();
    }

    #[test]
    fn test_take_sliced_large_list() {
        test_take_sliced_list_generic::<i64>();
    }

    #[test]
    fn test_take_sliced_list_with_value_nulls() {
        test_take_sliced_list_with_value_nulls_generic::<i32>();
    }

    #[test]
    fn test_take_sliced_large_list_with_value_nulls() {
        test_take_sliced_list_with_value_nulls_generic::<i64>();
    }

    #[test]
    fn test_take_runs() {
        let logical_array: Vec<i32> = vec![1_i32, 1, 2, 2, 1, 1, 1, 2, 2, 1, 1, 2, 2];

        let mut builder = PrimitiveRunBuilder::<Int32Type, Int32Type>::new();
        builder.extend(logical_array.into_iter().map(Some));
        let run_array = builder.finish();

        let take_indices: PrimitiveArray<Int32Type> =
            vec![7, 2, 3, 7, 11, 4, 6].into_iter().collect();

        let take_out = take_run(&run_array, &take_indices).unwrap();

        assert_eq!(take_out.len(), 7);
        assert_eq!(take_out.run_ends().len(), 7);
        assert_eq!(take_out.run_ends().values(), &[1_i32, 3, 4, 5, 7]);

        let take_out_values = take_out.values().as_primitive::<Int32Type>();
        assert_eq!(take_out_values.values(), &[2, 2, 2, 2, 1]);
    }

    #[test]
    fn test_take_runs_sliced() {
        let logical_array: Vec<i32> = vec![1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6];

        let mut builder = PrimitiveRunBuilder::<Int32Type, Int32Type>::new();
        builder.extend(logical_array.into_iter().map(Some));
        let run_array = builder.finish();

        let run_array = run_array.slice(4, 6); // [3, 3, 3, 4, 4, 5]

        let take_indices: PrimitiveArray<Int32Type> = vec![0, 5, 5, 1, 4].into_iter().collect();

        let result = take_run(&run_array, &take_indices).unwrap();
        let result = result.downcast::<Int32Array>().unwrap();

        let expected = vec![3, 5, 5, 3, 4];
        let actual = result.into_iter().flatten().collect::<Vec<_>>();

        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_value_index_from_fixed_list() {
        let list = FixedSizeListArray::from_iter_primitive::<Int32Type, _, _>(
            vec![
                Some(vec![Some(1), Some(2), None]),
                Some(vec![Some(4), None, Some(6)]),
                None,
                Some(vec![None, Some(8), Some(9)]),
            ],
            3,
        );

        let indices = UInt32Array::from(vec![2, 1, 0]);
        let indexed = take_value_indices_from_fixed_size_list(&list, &indices, 3).unwrap();

        assert_eq!(indexed, UInt32Array::from(vec![6, 7, 8, 3, 4, 5, 0, 1, 2]));

        let indices = UInt32Array::from(vec![3, 2, 1, 2, 0]);
        let indexed = take_value_indices_from_fixed_size_list(&list, &indices, 3).unwrap();

        assert_eq!(
            indexed,
            UInt32Array::from(vec![9, 10, 11, 6, 7, 8, 3, 4, 5, 6, 7, 8, 0, 1, 2])
        );
    }

    #[test]
    fn test_take_null_indices() {
        // Build indices with values that are out of bounds, but masked by null mask
        let indices = Int32Array::new(
            vec![1, 2, 400, 400].into(),
            Some(NullBuffer::from(vec![true, true, false, false])),
        );
        let values = Int32Array::from(vec![1, 23, 4, 5]);
        let r = take(&values, &indices, None).unwrap();
        let values = r
            .as_primitive::<Int32Type>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(&values, &[Some(23), Some(4), None, None])
    }

    #[test]
    fn test_take_fixed_size_list_null_indices() {
        let indices = Int32Array::from_iter([Some(0), None]);
        let values = Arc::new(Int32Array::from(vec![0, 1, 2, 3]));
        let arr_field = Arc::new(Field::new_list_field(values.data_type().clone(), true));
        let values = FixedSizeListArray::try_new(arr_field, 2, values, None).unwrap();

        let r = take(&values, &indices, None).unwrap();
        let values = r
            .as_fixed_size_list()
            .values()
            .as_primitive::<Int32Type>()
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(values, &[Some(0), Some(1), None, None])
    }

    #[test]
    fn test_take_bytes_null_indices() {
        let indices = Int32Array::new(
            vec![0, 1, 400, 400].into(),
            Some(NullBuffer::from_iter(vec![true, true, false, false])),
        );
        let values = StringArray::from(vec![Some("foo"), None]);
        let r = take(&values, &indices, None).unwrap();
        let values = r.as_string::<i32>().iter().collect::<Vec<_>>();
        assert_eq!(&values, &[Some("foo"), None, None, None])
    }

    #[test]
    fn test_take_union_sparse() {
        let structs = create_test_struct(vec![
            Some((Some(true), Some(42))),
            Some((Some(false), Some(28))),
            Some((Some(false), Some(19))),
            Some((Some(true), Some(31))),
            None,
        ]);
        let strings = StringArray::from(vec![Some("a"), None, Some("c"), None, Some("d")]);
        let type_ids = [1; 5].into_iter().collect::<ScalarBuffer<i8>>();

        let union_fields = [
            (
                0,
                Arc::new(Field::new("f1", structs.data_type().clone(), true)),
            ),
            (
                1,
                Arc::new(Field::new("f2", strings.data_type().clone(), true)),
            ),
        ]
        .into_iter()
        .collect();
        let children = vec![Arc::new(structs) as Arc<dyn Array>, Arc::new(strings)];
        let array = UnionArray::try_new(union_fields, type_ids, None, children).unwrap();

        let indices = vec![0, 3, 1, 0, 2, 4];
        let index = UInt32Array::from(indices.clone());
        let actual = take(&array, &index, None).unwrap();
        let actual = actual.as_any().downcast_ref::<UnionArray>().unwrap();
        let strings = actual.child(1);
        let strings = strings.as_any().downcast_ref::<StringArray>().unwrap();

        let actual = strings.iter().collect::<Vec<_>>();
        let expected = vec![Some("a"), None, None, Some("a"), Some("c"), Some("d")];
        assert_eq!(expected, actual);
    }

    #[test]
    fn test_take_union_dense() {
        let type_ids = vec![0, 1, 1, 0, 0, 1, 0];
        let offsets = vec![0, 0, 1, 1, 2, 2, 3];
        let ints = vec![10, 20, 30, 40];
        let strings = vec![Some("a"), None, Some("c"), Some("d")];

        let indices = vec![0, 3, 1, 0, 2, 4];

        let taken_type_ids = vec![0, 0, 1, 0, 1, 0];
        let taken_offsets = vec![0, 1, 0, 2, 1, 3];
        let taken_ints = vec![10, 20, 10, 30];
        let taken_strings = vec![Some("a"), None];

        let type_ids = <ScalarBuffer<i8>>::from(type_ids);
        let offsets = <ScalarBuffer<i32>>::from(offsets);
        let ints = UInt32Array::from(ints);
        let strings = StringArray::from(strings);

        let union_fields = [
            (
                0,
                Arc::new(Field::new("f1", ints.data_type().clone(), true)),
            ),
            (
                1,
                Arc::new(Field::new("f2", strings.data_type().clone(), true)),
            ),
        ]
        .into_iter()
        .collect();

        let array = UnionArray::try_new(
            union_fields,
            type_ids,
            Some(offsets),
            vec![Arc::new(ints), Arc::new(strings)],
        )
        .unwrap();

        let index = UInt32Array::from(indices);

        let actual = take(&array, &index, None).unwrap();
        let actual = actual.as_any().downcast_ref::<UnionArray>().unwrap();

        assert_eq!(actual.offsets(), Some(&ScalarBuffer::from(taken_offsets)));
        assert_eq!(actual.type_ids(), &ScalarBuffer::from(taken_type_ids));
        assert_eq!(
            UInt32Array::from(actual.child(0).to_data()),
            UInt32Array::from(taken_ints)
        );
        assert_eq!(
            StringArray::from(actual.child(1).to_data()),
            StringArray::from(taken_strings)
        );
    }

    #[test]
    fn test_take_union_dense_using_builder() {
        let mut builder = UnionBuilder::new_dense();

        builder.append::<Int32Type>("a", 1).unwrap();
        builder.append::<Float64Type>("b", 3.0).unwrap();
        builder.append::<Int32Type>("a", 4).unwrap();
        builder.append::<Int32Type>("a", 5).unwrap();
        builder.append::<Float64Type>("b", 2.0).unwrap();

        let union = builder.build().unwrap();

        let indices = UInt32Array::from(vec![2, 0, 1, 2]);

        let mut builder = UnionBuilder::new_dense();

        builder.append::<Int32Type>("a", 4).unwrap();
        builder.append::<Int32Type>("a", 1).unwrap();
        builder.append::<Float64Type>("b", 3.0).unwrap();
        builder.append::<Int32Type>("a", 4).unwrap();

        let taken = builder.build().unwrap();

        assert_eq!(
            taken.to_data(),
            take(&union, &indices, None).unwrap().to_data()
        );
    }

    #[test]
    fn test_take_union_dense_all_match_issue_6206() {
        let fields = UnionFields::from_fields(vec![Field::new("a", DataType::Int64, false)]);
        let ints = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5]));

        let array = UnionArray::try_new(
            fields,
            ScalarBuffer::from(vec![0_i8, 0, 0, 0, 0]),
            Some(ScalarBuffer::from_iter(0_i32..5)),
            vec![ints],
        )
        .unwrap();

        let indicies = Int64Array::from(vec![0, 2, 4]);
        let array = take(&array, &indicies, None).unwrap();
        assert_eq!(array.len(), 3);
    }

    #[test]
    fn test_take_bytes_offset_overflow() {
        let indices = Int32Array::from(vec![0; (i32::MAX >> 4) as usize]);
        let text = ('a'..='z').collect::<String>();
        let values = StringArray::from(vec![Some(text.clone())]);
        assert!(matches!(
            take(&values, &indices, None),
            Err(ArrowError::OffsetOverflowError(_))
        ));
    }

    #[test]
    fn test_take_run_empty_indices() {
        let mut builder = PrimitiveRunBuilder::<Int32Type, Int32Type>::new();
        builder.extend([Some(1), Some(1), Some(2), Some(2)]);
        let run_array = builder.finish();

        let logical_indices: PrimitiveArray<Int32Type> = PrimitiveArray::from(Vec::<i32>::new());

        let result = take_impl(&run_array, &logical_indices).expect("take_run with empty indices");

        // Verify the result is a valid empty RunArray
        assert_eq!(result.len(), 0);
        assert_eq!(result.null_count(), 0);

        // Verify that the result can be downcast and used without validation errors
        // This specifically tests that "The values in run_ends array should be strictly positive" is not triggered
        let run_result = result
            .as_any()
            .downcast_ref::<RunArray<Int32Type>>()
            .expect("result should be a RunArray");
        assert_eq!(run_result.run_ends().len(), 0);
        assert_eq!(run_result.values().len(), 0);
    }

    /// Sum of data-buffer capacities retained by a view array.
    fn view_buffer_capacity<T: ByteViewType>(v: &GenericByteViewArray<T>) -> usize {
        v.data_buffers().iter().map(|b| b.capacity()).sum()
    }

    /// A 40k-row string-view array whose values are long enough (> 12 bytes)
    /// to live in data buffers rather than inline in the views, and big
    /// enough overall to exceed the compaction skip gate.
    fn big_string_view() -> StringViewArray {
        (0..40_000)
            .map(|i| Some(format!("row-{i}-{}", "x".repeat(50))))
            .collect()
    }

    /// `take` of a few rows must not retain the input's full data buffers.
    /// Regression test for buffer-list compounding through chained hash
    /// joins (take clones the entire buffer list into every output batch).
    #[test]
    fn test_take_compacts_sparse_string_view() {
        let big = big_string_view();
        let taken = take(&big, &UInt32Array::from(vec![3u32, 7]), None).unwrap();
        let v = taken.as_string_view();
        assert_eq!(v.len(), 2);
        assert_eq!(v.value(0), big.value(3));
        assert_eq!(v.value(1), big.value(7));
        // Unreferenced buffers are pruned; what remains is bounded.
        assert!(
            v.data_buffers().len() <= 2,
            "{} buffers kept",
            v.data_buffers().len()
        );
        let capacity = view_buffer_capacity(v);
        assert!(
            capacity <= 1024 * 1024,
            "take output retains {capacity} buffer bytes for 2 rows"
        );
        // Negative control: a dense take (all rows) must NOT copy.
        let all: UInt32Array = (0..40_000u32).collect();
        let dense = take(&big, &all, None).unwrap();
        let dense = dense.as_string_view();
        // dense output reuses the input's buffers zero-copy (no gc copy)
        assert_eq!(dense.data_buffers().len(), big.data_buffers().len());
        assert_eq!(
            dense.data_buffers()[0].as_ptr(),
            big.data_buffers()[0].as_ptr()
        );
    }

    /// `concat` of sparse slices must not retain every input's full buffer
    /// list (the CollectLeft build-side collect path).
    #[test]
    fn test_concat_compacts_sparse_string_view() {
        use crate::concat::concat;
        let big = big_string_view();
        let a = big.slice(0, 2);
        let b = big.slice(5, 2);
        let out = concat(&[&a, &b]).unwrap();
        let v = out.as_string_view();
        assert_eq!(v.len(), 4);
        assert_eq!(v.value(0), big.value(0));
        assert_eq!(v.value(2), big.value(5));
        assert!(
            v.data_buffers().len() <= 2,
            "{} buffers kept",
            v.data_buffers().len()
        );
        let capacity = view_buffer_capacity(v);
        assert!(
            capacity <= 1024 * 1024,
            "concat output retains {capacity} buffer bytes for 4 rows"
        );
    }

    /// Views nested inside container types (here map<string,string>) take a
    /// different code path (`MutableArrayData`) than top-level views and
    /// must also be compacted.
    #[test]
    fn test_take_compacts_sparse_views_nested_in_map() {
        let mut builder = MapBuilder::new(None, StringViewBuilder::new(), StringViewBuilder::new());
        for i in 0..20_000 {
            builder
                .keys()
                .append_value(format!("key-{i}-{}", "k".repeat(50)));
            builder
                .values()
                .append_value(format!("val-{i}-{}", "v".repeat(50)));
            builder.append(true).unwrap();
        }
        let map = builder.finish();
        let taken = take(&map, &UInt32Array::from(vec![3u32, 7]), None).unwrap();
        let entries = taken.as_map().entries();
        let keys = entries.column(0).as_string_view();
        assert_eq!(keys.value(0), format!("key-3-{}", "k".repeat(50)));
        assert_eq!(keys.value(1), format!("key-7-{}", "k".repeat(50)));
        for col in entries.columns() {
            let v = col.as_string_view();
            let capacity = view_buffer_capacity(v);
            assert!(
                capacity <= 1024 * 1024,
                "nested view column retains {capacity} buffer bytes for 2 rows"
            );
        }
    }

    /// Top-level views already prune inside `interleave_views`, so this pins
    /// the properties the added compaction pass must not break: values stay
    /// correct, and a dense selection still shares its source's buffers
    /// rather than paying a copy.
    #[test]
    fn test_interleave_compacts_sparse_string_view() {
        use crate::interleave::interleave;
        let a = big_string_view();
        let b = big_string_view();
        let out = interleave(&[&a, &b], &[(0, 3), (1, 7), (0, 11)]).unwrap();
        let v = out.as_string_view();
        assert_eq!(v.len(), 3);
        assert_eq!(v.value(0), a.value(3));
        assert_eq!(v.value(1), b.value(7));
        assert_eq!(v.value(2), a.value(11));
        let capacity = view_buffer_capacity(v);
        assert!(
            capacity <= 1024 * 1024,
            "interleave output retains {capacity} buffer bytes for 3 rows"
        );
        // Negative control: taking every row from one source references all
        // of its buffers, so there is nothing to prune and nothing is copied.
        let all: Vec<(usize, usize)> = (0..a.len()).map(|i| (0, i)).collect();
        let dense = interleave(&[&a, &b], &all).unwrap();
        let dense = dense.as_string_view();
        assert_eq!(dense.data_buffers().len(), a.data_buffers().len());
        assert_eq!(dense.data_buffers()[0].as_ptr(), a.data_buffers()[0].as_ptr());
    }

    /// `Map` is absent from `interleave`'s dispatch, so it falls to
    /// `interleave_fallback` -> `MutableArrayData`, which keeps the inputs'
    /// entire buffer lists. Nested view columns must compact there too.
    #[test]
    fn test_interleave_compacts_sparse_views_nested_in_map() {
        use crate::interleave::interleave;
        let mut builder = MapBuilder::new(None, StringViewBuilder::new(), StringViewBuilder::new());
        for i in 0..20_000 {
            builder
                .keys()
                .append_value(format!("key-{i}-{}", "k".repeat(50)));
            builder
                .values()
                .append_value(format!("val-{i}-{}", "v".repeat(50)));
            builder.append(true).unwrap();
        }
        let map = builder.finish();
        let out = interleave(&[&map, &map], &[(0, 3), (1, 7)]).unwrap();
        let entries = out.as_map().entries();
        let keys = entries.column(0).as_string_view();
        assert_eq!(keys.value(0), format!("key-3-{}", "k".repeat(50)));
        assert_eq!(keys.value(1), format!("key-7-{}", "k".repeat(50)));
        for col in entries.columns() {
            let v = col.as_string_view();
            let capacity = view_buffer_capacity(v);
            assert!(
                capacity <= 1024 * 1024,
                "nested view column retains {capacity} buffer bytes for 2 rows"
            );
        }
    }

    /// `concat` of clones of one array must dedup the repeated buffers with
    /// no data copy (the CollectLeft build-side collect shape).
    #[test]
    fn test_concat_dedups_repeated_buffers_without_copy() {
        use crate::concat::concat;
        let big = big_string_view();
        let inputs: Vec<&dyn Array> = (0..8).map(|_| &big as &dyn Array).collect();
        let out = concat(&inputs).unwrap();
        let v = out.as_string_view();
        assert_eq!(v.len(), 320_000);
        assert_eq!(v.value(3), big.value(3));
        assert_eq!(v.value(280_007), big.value(7));
        // the same buffers, each kept once, not copied
        assert_eq!(v.data_buffers().len(), big.data_buffers().len());
        assert_eq!(v.data_buffers()[0].as_ptr(), big.data_buffers()[0].as_ptr());
    }

    /// `take` on a list-view shares the full values child, whose own views
    /// reference every buffer, so there is nothing to prune at the leaf
    /// (reachability through the parent's offsets is not projected). The
    /// shared child also means the buffer list cannot grow through take;
    /// this pins values staying correct and retention not growing.
    #[test]
    fn test_take_on_list_view_keeps_values_correct_without_growth() {
        let mut builder = ListViewBuilder::new(StringViewBuilder::new());
        for i in 0..20_000 {
            builder
                .values()
                .append_value(format!("row-{i}-{}", "x".repeat(50)));
            builder.append(true);
        }
        let list = builder.finish();
        let input_capacity = view_buffer_capacity(list.values().as_string_view());
        let taken = take(&list, &UInt32Array::from(vec![3u32, 7]), None).unwrap();
        let taken = taken.as_list_view::<i32>();
        assert_eq!(
            taken.value(0).as_string_view().value(0),
            format!("row-3-{}", "x".repeat(50))
        );
        let values = taken.values().as_string_view();
        assert!(view_buffer_capacity(values) <= input_capacity);
    }

    /// Null map slots must survive compaction intact.
    #[test]
    fn test_take_compacts_map_with_null_slots() {
        let mut builder = MapBuilder::new(None, StringViewBuilder::new(), StringViewBuilder::new());
        for i in 0..10_000 {
            if i % 3 == 0 {
                builder.append(false).unwrap();
            } else {
                builder
                    .keys()
                    .append_value(format!("key-{i}-{}", "k".repeat(50)));
                builder
                    .values()
                    .append_value(format!("val-{i}-{}", "v".repeat(50)));
                builder.append(true).unwrap();
            }
        }
        let map = builder.finish();
        let taken = take(&map, &UInt32Array::from(vec![3u32, 4, 6]), None).unwrap();
        let m = taken.as_map();
        assert!(m.is_null(0), "row 3 was a null map slot");
        assert!(m.is_null(2), "row 6 was a null map slot");
        let entries = m.value(1);
        let keys = entries.column(0).as_string_view();
        assert_eq!(keys.value(0), format!("key-4-{}", "k".repeat(50)));
    }

    /// A single huge buffer that only a few views reference must be copied
    /// down (the prune stage alone cannot shrink it).
    #[test]
    fn test_take_copies_when_pruning_cannot_shrink() {
        // gc() the builder output first so all values live in one buffer.
        let big: StringViewArray = (0..40_000)
            .map(|i| Some(format!("row-{i}-{}", "x".repeat(50))))
            .collect();
        let big = big.gc();
        assert_eq!(big.data_buffers().len(), 1);
        let taken = take(&big, &UInt32Array::from(vec![3u32, 7]), None).unwrap();
        let v = taken.as_string_view();
        assert_eq!(v.value(0), big.value(3));
        let capacity = view_buffer_capacity(v);
        let used = v.total_buffer_bytes_used();
        assert!(
            capacity <= used * 2,
            "sparse take of one huge buffer retains {capacity} bytes for {used} referenced"
        );
    }

    /// Two distinct surviving buffers must keep their views pointing at the
    /// right one after duplicates are pruned (guards the index remap).
    #[test]
    fn test_concat_remap_preserves_values_across_surviving_buffers() {
        use crate::concat::concat;
        let a: StringViewArray = (0..12_000)
            .map(|i| Some(format!("a-{i}-{}", "a".repeat(50))))
            .collect();
        let a = a.gc();
        let b: StringViewArray = (0..12_000)
            .map(|i| Some(format!("b-{i}-{}", "b".repeat(50))))
            .collect();
        let b = b.gc();
        // a appears twice: its buffer entry must dedup, b's must survive,
        // and every view must land on the right buffer afterwards.
        let out = concat(&[&a, &b, &a]).unwrap();
        let v = out.as_string_view();
        assert_eq!(v.data_buffers().len(), 2);
        assert_eq!(v.value(3), a.value(3));
        assert_eq!(v.value(12_003), b.value(3));
        assert_eq!(v.value(24_003), a.value(3));
    }

    /// Null slots must survive the prune/remap path.
    #[test]
    fn test_take_compaction_preserves_nulls_in_view_column() {
        let big: StringViewArray = (0..40_000)
            .map(|i| {
                if i % 2 == 0 {
                    Some(format!("row-{i}-{}", "x".repeat(50)))
                } else {
                    None
                }
            })
            .collect();
        let taken = take(&big, &UInt32Array::from(vec![2u32, 3, 4]), None).unwrap();
        let v = taken.as_string_view();
        assert_eq!(v.value(0), big.value(2));
        assert!(v.is_null(1));
        assert_eq!(v.value(2), big.value(4));
        assert!(view_buffer_capacity(v) <= 1024 * 1024);
    }

    /// Union children are taken per child without compaction, so the deep
    /// pass must descend into unions too.
    #[test]
    fn test_take_compacts_sparse_views_nested_in_union() {
        use std::sync::Arc;
        let strings: StringViewArray = (0..20_000)
            .map(|i| Some(format!("row-{i}-{}", "x".repeat(50))))
            .collect();
        let fields = [(0i8, Arc::new(Field::new("s", DataType::Utf8View, false)))]
            .into_iter()
            .collect::<UnionFields>();
        let type_ids = vec![0i8; 20_000].into();
        let union =
            UnionArray::try_new(fields, type_ids, None, vec![Arc::new(strings.clone())]).unwrap();
        let taken = take(&union, &UInt32Array::from(vec![3u32, 7]), None).unwrap();
        let u = taken.as_union();
        assert_eq!(u.value(0).as_string_view().value(0), strings.value(3));
        let child = u.child(0).as_string_view();
        let capacity = view_buffer_capacity(child);
        assert!(
            capacity <= 1024 * 1024,
            "union view child retains {capacity} buffer bytes for 2 rows"
        );
    }

    /// Reproduces the compounding this change bounds, and prints the numbers.
    ///
    /// Each level mimics one hash-join level: `concat` merges the buffer lists
    /// of 16 probe-partition batches, then `take` selects the same small row
    /// count back out and clones the merged list into the output. The row count
    /// is constant across levels, so anything that grows is pure retention.
    ///
    /// ```text
    /// cargo test -p arrow-select --release compounding -- --ignored --nocapture
    ///
    /// level 0: 64 rows,  8 buffers, 4177920 bytes retained
    /// level 1: 64 rows,  1 buffers,   16384 bytes retained
    /// level 2: 64 rows, 16 buffers,  262144 bytes retained
    /// level 3: 64 rows,  1 buffers,   16384 bytes retained
    /// level 4: 64 rows, 16 buffers,  262144 bytes retained
    /// ```
    ///
    /// The saw-tooth is the skip gate: a merged list within both
    /// `COMPACT_MIN_CAPACITY` and `COMPACT_MAX_SKIP_BUFFERS` is left alone, and
    /// the next level's merge crosses the gate and collapses back to one buffer.
    ///
    /// For the unfixed baseline, drop the `deep_compact_views` call at the end
    /// of `take()` and `concat()` and raise the bound asserted below. That gives
    /// 128 / 2048 / 32768 buffers and 66.8 MB / 1.07 GB / 17.1 GB over three
    /// levels. Stop at three: the figure is buffer-list capacity, which the
    /// harness reaches by re-merging one source array, and a real plan whose
    /// partitions hold distinct buffers pays it in resident memory.
    #[test]
    #[ignore = "measurement harness, prints per-level retention"]
    fn compounding_across_levels_stays_bounded() {
        const LEVELS: usize = 4;
        const FANOUT: usize = 16;

        let source = big_string_view();
        let mut cur: ArrayRef = Arc::new(source.slice(0, 64));
        let rows = cur.len();
        let indices: UInt32Array = (0..rows as u32).collect();

        println!(
            "level 0: {rows} rows, {} buffers, {} bytes retained",
            cur.as_string_view().data_buffers().len(),
            view_buffer_capacity(cur.as_string_view())
        );

        for level in 1..=LEVELS {
            let inputs = vec![cur.as_ref(); FANOUT];
            let merged = crate::concat::concat(&inputs).unwrap();
            cur = take(&merged, &indices, None).unwrap();

            let v = cur.as_string_view();
            let (buffers, bytes) = (v.data_buffers().len(), view_buffer_capacity(v));
            println!(
                "level {level}: {} rows, {buffers} buffers, {bytes} bytes retained",
                v.len()
            );

            assert_eq!(v.len(), rows, "row count must stay constant");
            assert_eq!(v.value(0), source.value(0));
            assert!(
                bytes <= 4 * 1024 * 1024,
                "level {level} retains {bytes} bytes for {rows} rows"
            );
        }
    }
}
