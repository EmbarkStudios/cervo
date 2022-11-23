// Author: Tom Solberg <tom.solberg@embark-studios.com>
// Copyright © 2022, Embark Studios, all rights reserved.
// Created: 27 July 2022

use std::{cell::RefCell, collections::HashMap};

use cervo_core::prelude::{Batcher, Inferer, InfererExt, State};

struct TestInferer<
    B: Fn(usize) -> usize,
    R: Fn(cervo_core::batcher::ScratchPadView<'_>) -> anyhow::Result<(), anyhow::Error>,
> {
    batch_size: B,
    raw: R,
    in_shapes: Vec<(String, Vec<usize>)>,
    out_shapes: Vec<(String, Vec<usize>)>,
}

impl<B, R> Inferer for TestInferer<B, R>
where
    B: Fn(usize) -> usize,
    R: Fn(cervo_core::batcher::ScratchPadView<'_>) -> anyhow::Result<(), anyhow::Error>,
{
    fn select_batch_size(&self, max_count: usize) -> usize {
        (self.batch_size)(max_count)
    }

    fn infer_raw(
        &self,
        batch: cervo_core::batcher::ScratchPadView<'_>,
    ) -> anyhow::Result<(), anyhow::Error> {
        (self.raw)(batch)
    }

    fn input_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.in_shapes
    }

    fn output_shapes(&self) -> &[(String, Vec<usize>)] {
        &self.out_shapes
    }
}

#[test]
fn test_construct_wrapper() {
    let inf = TestInferer {
        batch_size: |_| 3,
        raw: |_| Ok(()),
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let _batched = inf.into_batched();
}

#[test]
fn test_construct_loose() {
    let inf = TestInferer {
        batch_size: |_| 3,
        raw: |_| Ok(()),
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let _batcher = Batcher::new(&inf);
}

#[test]
fn test_push_basic() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 1,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 1);
            assert_eq!(b.input_slot(0).len(), 11);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = Batcher::new(&inf);

    for id in 0..2 {
        let mut s = State::empty();
        s.data.insert("first", vec![0.0; 11]);
        batcher.push(id, s).unwrap();
    }

    batcher.execute(&inf).unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_push_wrapped() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 1,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 1);
            assert_eq!(b.input_slot(0).len(), 11);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = inf.into_batched();

    for id in 0..2 {
        let s = State::empty();
        batcher.push(id, s).unwrap();
    }

    batcher.execute().unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_push_two() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 2,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 2);
            assert_eq!(b.input_slot(0).len(), 22);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = Batcher::new(&inf);

    for id in 0..4 {
        let mut s = State::empty();
        s.data.insert("first", vec![0.0; 11]);
        batcher.push(id, s).unwrap();
    }

    batcher.execute(&inf).unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_extend_single() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 1,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 1);
            assert_eq!(b.input_slot(0).len(), 11);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = Batcher::new(&inf);

    let mut batch: HashMap<u64, State<'static>> = HashMap::default();

    let first = &"first";
    for id in 0..2 {
        let mut s = State::empty();
        s.data.insert(first, vec![0.0; 11]);
        batch.insert(id, s);
    }

    batcher.extend(batch).unwrap();
    batcher.execute(&inf).unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_extend_wrapped() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 1,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 1);
            assert_eq!(b.input_slot(0).len(), 11);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = inf.into_batched();

    let mut batch: HashMap<u64, State<'static>> = HashMap::default();

    let first = &"first";
    for id in 0..2 {
        let mut s = State::empty();
        s.data.insert(first, vec![0.0; 11]);
        batch.insert(id, s);
    }

    batcher.extend(batch).unwrap();
    batcher.execute().unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_extend_double() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 2,
        raw: |b| {
            *call_count.borrow_mut() += 1;
            assert_eq!(b.len(), 2);
            assert_eq!(b.input_slot(0).len(), 22);
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = Batcher::new(&inf);

    let mut batch: HashMap<u64, State<'static>> = HashMap::default();

    let first = "first";
    for id in 0..4 {
        let mut s = State::empty();
        s.data.insert(first, vec![0.0; 11]);
        batch.insert(id, s);
    }

    batcher.extend(batch).unwrap();
    batcher.execute(&inf).unwrap();
    assert_eq!(call_count.take(), 2);
}

#[test]
fn test_values() {
    let call_count = RefCell::new(0);
    let inf = TestInferer {
        batch_size: |_| 2,
        raw: |mut b| {
            assert_eq!(b.len(), 2);
            assert_eq!(b.input_slot(0).len(), 22);
            assert_eq!(
                b.input_slot(0),
                (11 * (*call_count.borrow() * b.len())
                    ..(*call_count.borrow() * b.len() + b.len()) * 11)
                    .map(|i| i as f32)
                    .collect::<Vec<_>>()
            );
            let l = b.len();
            let out = b.output_slot_mut(0);
            for (i, o) in out.iter_mut().enumerate() {
                *o = (*call_count.borrow() * l) as f32 + i as f32 / 11.0;
            }
            *call_count.borrow_mut() += 1;
            Ok(())
        },
        in_shapes: vec![("first".to_owned(), vec![11])],
        out_shapes: vec![("out".to_owned(), vec![11])],
    };

    let mut batcher = Batcher::new(&inf);

    let first = &"first";
    for id in 0..4 {
        let mut s = State::empty();
        s.data
            .insert(first, (11 * id..(id + 1) * 11).map(|i| i as f32).collect());

        batcher.push(id, s).unwrap();
    }

    let r = batcher.execute(&inf).unwrap();
    assert_eq!(r.len(), 4);
    let _ = &r;
    for (id, vals) in r {
        assert_eq!(vals.data["out"].len(), 11);
        assert_eq!(vals.data["out"][0], id as f32);
    }

    assert_eq!(call_count.take(), 2);
}
