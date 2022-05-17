#!/usr/bin/env bash
export FAKETIME="2020-12-24 20:30:00"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/faketime/libfaketime.so.1

cargo run batch-to-nnef *.onnx
cargo run package test.onnx test
cargo run package -O test.onnx test-nnef
