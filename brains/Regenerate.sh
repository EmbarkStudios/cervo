#!/usr/bin/env bash

# can't build with faketime

cargo build

export FAKETIME="2020-12-24 20:30:00"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1

cargo run batch-to-nnef *.onnx

cargo run package test-large.onnx test-large
cargo run package -C test-large.onnx test-large-c

cargo run package -O test-large.onnx test-large-nnef
cargo run package -C -O test-large.onnx test-large-nnef-c
