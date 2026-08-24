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

//! Prices the data-buffer bookkeeping that `take`, `filter`, `concat` and
//! `interleave` do for view arrays.
//!
//! Each kernel decides which of its inputs' variadic data buffers the output
//! needs, and rewrites the surviving views' buffer indices to match. The work
//! is one pass over the output views plus one hash lookup per distinct buffer,
//! so the cases below vary the two things it scales with, the number of views
//! and the number of buffers, and separate the paths that reach it.
//!
//! Every case is meant to be compared against the same case on another build,
//! never against a sibling case: the inputs differ in size between cases by
//! design, so cross-case ratios mean nothing.
//!
//! The shapes:
//!
//! * `flat-few` and `flat-many`: a `Utf8View` column whose values sit in one
//!   large data buffer, versus the same values spread over many small ones.
//!   The gap between the two is the per-buffer cost.
//! * `nested`: a `map<utf8view, utf8view>` column. Every kernel reaches this
//!   through `MutableArrayData` rather than its own view path, so this and
//!   `flat-*` price different code.
//! * `wide`: a record batch of many `Utf8View` columns through the batch entry
//!   points. Per-column fixed cost is multiplied by the column count here, so
//!   a cost invisible on one narrow column shows up in this row.
//! * `primitive`: an `Int64` column, which has no views and no data buffers.
//!   This prices only the type check on arrays the bookkeeping never touches.
//!
//! `dense` keeps every row, which is the case that should cost nothing beyond
//! what the kernel already did. `sparse` keeps one row in 64, which is where
//! the pruning has real work to do.

#[macro_use]
extern crate criterion;

use criterion::Criterion;

extern crate arrow;

use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema};
use arrow_select::concat::{concat, concat_batches};
use arrow_select::filter::{filter, filter_record_batch};
use arrow_select::interleave::{interleave, interleave_record_batch};
use arrow_select::take::{take, take_record_batch};
use std::hint::black_box;
use std::sync::Arc;

/// Rows per input array. One batch's worth, so the numbers read as per-batch
/// kernel cost.
const ROWS: usize = 8192;
/// Value length, over the 12 bytes that would be inlined into the view and so
/// reference no buffer at all.
const VALUE_LEN: usize = 32;
/// Kept rows per this many, for the `sparse` cases.
const SPARSE_STRIDE: usize = 64;
/// Columns in the `wide` cases.
const WIDE_COLUMNS: usize = 32;
/// Block size that puts every value in its own data buffer.
const BLOCK_PER_VALUE: u32 = VALUE_LEN as u32;
/// Block size large enough to hold every value in one data buffer.
const BLOCK_SINGLE: u32 = (ROWS * VALUE_LEN) as u32;

fn view_array(rows: usize, block_size: u32) -> StringViewArray {
    let mut builder = StringViewBuilder::new().with_fixed_block_size(block_size);
    for i in 0..rows {
        builder.append_value(format!("{i:0>VALUE_LEN$}"));
    }
    builder.finish()
}

fn map_of_views(rows: usize, block_size: u32) -> MapArray {
    let mut builder = MapBuilder::new(
        None,
        StringViewBuilder::new().with_fixed_block_size(block_size),
        StringViewBuilder::new().with_fixed_block_size(block_size),
    );
    for i in 0..rows {
        builder.keys().append_value(format!("k{i:0>VALUE_LEN$}"));
        builder.values().append_value(format!("v{i:0>VALUE_LEN$}"));
        builder.append(true).unwrap();
    }
    builder.finish()
}

fn wide_batch(rows: usize, columns: usize, block_size: u32) -> RecordBatch {
    let fields: Vec<Field> = (0..columns)
        .map(|c| Field::new(format!("c{c}"), DataType::Utf8View, false))
        .collect();
    let arrays: Vec<ArrayRef> = (0..columns)
        .map(|_| Arc::new(view_array(rows, block_size)) as ArrayRef)
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap()
}

/// Every row, in order: the kernel copies as much as it reads.
fn dense_indices(rows: usize) -> UInt32Array {
    UInt32Array::from_iter_values(0..rows as u32)
}

/// One row per [`SPARSE_STRIDE`], spread across the input so the surviving rows
/// have no relationship to the buffers their values happen to live in.
fn sparse_indices(rows: usize) -> UInt32Array {
    UInt32Array::from_iter_values((0..rows as u32).step_by(SPARSE_STRIDE))
}

fn dense_mask(rows: usize) -> BooleanArray {
    BooleanArray::from_iter((0..rows).map(|_| Some(true)))
}

fn sparse_mask(rows: usize) -> BooleanArray {
    BooleanArray::from_iter((0..rows).map(|i| Some(i % SPARSE_STRIDE == 0)))
}

/// `(array index, row index)` pairs drawn round robin from `inputs` arrays,
/// which is what makes buffers shared between inputs reachable twice.
fn interleave_indices(inputs: usize, rows: usize, stride: usize) -> Vec<(usize, usize)> {
    (0..rows)
        .step_by(stride)
        .enumerate()
        .map(|(n, row)| (n % inputs, row))
        .collect()
}

/// Inputs for the `dense` concat cases: whole arrays, no buffer shared between
/// them.
fn concat_whole(array: &StringViewArray, inputs: usize) -> Vec<ArrayRef> {
    (0..inputs)
        .map(|_| Arc::new(array.clone()) as ArrayRef)
        .collect()
}

/// Inputs for the `sparse` concat cases: thin slices of one array, so every
/// input carries the whole source buffer list while reaching almost none of it.
/// This is the shape a fanout of selections over one source produces.
fn concat_slices(array: &StringViewArray, slice_rows: usize) -> Vec<ArrayRef> {
    (0..array.len())
        .step_by(SPARSE_STRIDE)
        .map(|start| {
            let len = slice_rows.min(array.len() - start);
            Arc::new(array.slice(start, len)) as ArrayRef
        })
        .collect()
}

fn bench_take(c: &mut Criterion) {
    for (shape, block) in [("flat-few", BLOCK_SINGLE), ("flat-many", BLOCK_PER_VALUE)] {
        let array = view_array(ROWS, block);
        let dense = dense_indices(ROWS);
        let sparse = sparse_indices(ROWS);
        c.bench_function(&format!("take {shape} dense"), |b| {
            b.iter(|| black_box(take(&array, &dense, None).unwrap()))
        });
        c.bench_function(&format!("take {shape} sparse"), |b| {
            b.iter(|| black_box(take(&array, &sparse, None).unwrap()))
        });
    }

    let nested = map_of_views(ROWS, BLOCK_PER_VALUE);
    let dense = dense_indices(ROWS);
    let sparse = sparse_indices(ROWS);
    c.bench_function("take nested dense", |b| {
        b.iter(|| black_box(take(&nested, &dense, None).unwrap()))
    });
    c.bench_function("take nested sparse", |b| {
        b.iter(|| black_box(take(&nested, &sparse, None).unwrap()))
    });

    let wide = wide_batch(ROWS, WIDE_COLUMNS, BLOCK_PER_VALUE);
    c.bench_function("take wide dense", |b| {
        b.iter(|| black_box(take_record_batch(&wide, &dense).unwrap()))
    });
    c.bench_function("take wide sparse", |b| {
        b.iter(|| black_box(take_record_batch(&wide, &sparse).unwrap()))
    });

    let primitive = Int64Array::from_iter_values(0..ROWS as i64);
    c.bench_function("take primitive dense", |b| {
        b.iter(|| black_box(take(&primitive, &dense, None).unwrap()))
    });
}

fn bench_filter(c: &mut Criterion) {
    for (shape, block) in [("flat-few", BLOCK_SINGLE), ("flat-many", BLOCK_PER_VALUE)] {
        let array = view_array(ROWS, block);
        let dense = dense_mask(ROWS);
        let sparse = sparse_mask(ROWS);
        c.bench_function(&format!("filter {shape} dense"), |b| {
            b.iter(|| black_box(filter(&array, &dense).unwrap()))
        });
        c.bench_function(&format!("filter {shape} sparse"), |b| {
            b.iter(|| black_box(filter(&array, &sparse).unwrap()))
        });
    }

    let nested = map_of_views(ROWS, BLOCK_PER_VALUE);
    let dense = dense_mask(ROWS);
    let sparse = sparse_mask(ROWS);
    c.bench_function("filter nested dense", |b| {
        b.iter(|| black_box(filter(&nested, &dense).unwrap()))
    });
    c.bench_function("filter nested sparse", |b| {
        b.iter(|| black_box(filter(&nested, &sparse).unwrap()))
    });

    let wide = wide_batch(ROWS, WIDE_COLUMNS, BLOCK_PER_VALUE);
    c.bench_function("filter wide dense", |b| {
        b.iter(|| black_box(filter_record_batch(&wide, &dense).unwrap()))
    });
    c.bench_function("filter wide sparse", |b| {
        b.iter(|| black_box(filter_record_batch(&wide, &sparse).unwrap()))
    });

    let primitive = Int64Array::from_iter_values(0..ROWS as i64);
    c.bench_function("filter primitive dense", |b| {
        b.iter(|| black_box(filter(&primitive, &dense).unwrap()))
    });
}

fn bench_concat(c: &mut Criterion) {
    /// Inputs per dense concat case.
    const WHOLE_INPUTS: usize = 4;
    /// Rows per slice in the sparse concat cases.
    const SLICE_ROWS: usize = 4;

    for (shape, block) in [("flat-few", BLOCK_SINGLE), ("flat-many", BLOCK_PER_VALUE)] {
        let array = view_array(ROWS, block);
        let whole = concat_whole(&array, WHOLE_INPUTS);
        let whole: Vec<&dyn Array> = whole.iter().map(|a| a.as_ref()).collect();
        let slices = concat_slices(&array, SLICE_ROWS);
        let slices: Vec<&dyn Array> = slices.iter().map(|a| a.as_ref()).collect();
        c.bench_function(&format!("concat {shape} dense"), |b| {
            b.iter(|| black_box(concat(&whole).unwrap()))
        });
        c.bench_function(&format!("concat {shape} sparse"), |b| {
            b.iter(|| black_box(concat(&slices).unwrap()))
        });
    }

    let nested = map_of_views(ROWS, BLOCK_PER_VALUE);
    let whole: Vec<ArrayRef> = (0..WHOLE_INPUTS)
        .map(|_| Arc::new(nested.clone()) as ArrayRef)
        .collect();
    let whole_refs: Vec<&dyn Array> = whole.iter().map(|a| a.as_ref()).collect();
    let slices: Vec<ArrayRef> = (0..nested.len())
        .step_by(SPARSE_STRIDE)
        .map(|start| {
            Arc::new(nested.slice(start, SLICE_ROWS.min(nested.len() - start))) as ArrayRef
        })
        .collect();
    let slice_refs: Vec<&dyn Array> = slices.iter().map(|a| a.as_ref()).collect();
    c.bench_function("concat nested dense", |b| {
        b.iter(|| black_box(concat(&whole_refs).unwrap()))
    });
    c.bench_function("concat nested sparse", |b| {
        b.iter(|| black_box(concat(&slice_refs).unwrap()))
    });

    let wide = wide_batch(ROWS, WIDE_COLUMNS, BLOCK_PER_VALUE);
    let schema = wide.schema();
    let whole_batches: Vec<RecordBatch> = (0..WHOLE_INPUTS).map(|_| wide.clone()).collect();
    let slice_batches: Vec<RecordBatch> = (0..wide.num_rows())
        .step_by(SPARSE_STRIDE)
        .map(|start| wide.slice(start, SLICE_ROWS.min(wide.num_rows() - start)))
        .collect();
    c.bench_function("concat wide dense", |b| {
        b.iter(|| black_box(concat_batches(&schema, &whole_batches).unwrap()))
    });
    c.bench_function("concat wide sparse", |b| {
        b.iter(|| black_box(concat_batches(&schema, &slice_batches).unwrap()))
    });

    let primitive: ArrayRef = Arc::new(Int64Array::from_iter_values(0..ROWS as i64));
    let primitives: Vec<&dyn Array> = (0..WHOLE_INPUTS).map(|_| primitive.as_ref()).collect();
    c.bench_function("concat primitive dense", |b| {
        b.iter(|| black_box(concat(&primitives).unwrap()))
    });
}

fn bench_interleave(c: &mut Criterion) {
    /// Inputs per interleave case, all slices of one source, so the same
    /// allocation is reachable under several input indices.
    const INPUTS: usize = 4;

    let dense_idx = interleave_indices(INPUTS, ROWS, 1);
    let sparse_idx = interleave_indices(INPUTS, ROWS, SPARSE_STRIDE);

    for (shape, block) in [("flat-few", BLOCK_SINGLE), ("flat-many", BLOCK_PER_VALUE)] {
        let array = view_array(ROWS, block);
        let inputs: Vec<ArrayRef> = (0..INPUTS)
            .map(|_| Arc::new(array.clone()) as ArrayRef)
            .collect();
        let refs: Vec<&dyn Array> = inputs.iter().map(|a| a.as_ref()).collect();
        c.bench_function(&format!("interleave {shape} dense"), |b| {
            b.iter(|| black_box(interleave(&refs, &dense_idx).unwrap()))
        });
        c.bench_function(&format!("interleave {shape} sparse"), |b| {
            b.iter(|| black_box(interleave(&refs, &sparse_idx).unwrap()))
        });
    }

    let nested = map_of_views(ROWS, BLOCK_PER_VALUE);
    let inputs: Vec<ArrayRef> = (0..INPUTS)
        .map(|_| Arc::new(nested.clone()) as ArrayRef)
        .collect();
    let refs: Vec<&dyn Array> = inputs.iter().map(|a| a.as_ref()).collect();
    c.bench_function("interleave nested dense", |b| {
        b.iter(|| black_box(interleave(&refs, &dense_idx).unwrap()))
    });
    c.bench_function("interleave nested sparse", |b| {
        b.iter(|| black_box(interleave(&refs, &sparse_idx).unwrap()))
    });

    let wide = wide_batch(ROWS, WIDE_COLUMNS, BLOCK_PER_VALUE);
    let batches: Vec<RecordBatch> = (0..INPUTS).map(|_| wide.clone()).collect();
    let batch_refs: Vec<&RecordBatch> = batches.iter().collect();
    c.bench_function("interleave wide dense", |b| {
        b.iter(|| black_box(interleave_record_batch(&batch_refs, &dense_idx).unwrap()))
    });
    c.bench_function("interleave wide sparse", |b| {
        b.iter(|| black_box(interleave_record_batch(&batch_refs, &sparse_idx).unwrap()))
    });

    let primitive: ArrayRef = Arc::new(Int64Array::from_iter_values(0..ROWS as i64));
    let primitives: Vec<&dyn Array> = (0..INPUTS).map(|_| primitive.as_ref()).collect();
    c.bench_function("interleave primitive dense", |b| {
        b.iter(|| black_box(interleave(&primitives, &dense_idx).unwrap()))
    });
}

criterion_group!(
    benches,
    bench_take,
    bench_filter,
    bench_concat,
    bench_interleave
);
criterion_main!(benches);
