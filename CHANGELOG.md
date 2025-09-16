<!-- markdownlint-disable blanks-around-headings blanks-around-lists no-duplicate-heading -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->
## [Unreleased] - ReleaseDate
## [0.9.2] - 2025-09-16
- Fix bugs in the new wrapper setup where consumed and modified shapes
  weren't respected during wrapper construction.

## [0.9.1] - 2025-09-10

- Add `StatefulInferer::replace_inferer` which works with a `&mut
  StatefulInferer`, at the cost of requiring the inferer to be of the
  same type.
- Fix bugs in the new wrapper setup where consumed and modified shapes
  weren't respected during wrapper construction.

## [0.9.0] - 2025-09-04

- Breaking: `Inferer.begin_agent` and `Inferer.end_agent` now take
  `&self`, changed from a mutable reference.

### Wrapper rework

To support a wider variety of uses, we have implemented a new category
of wrappers that do not require ownership of the inferer. This allows
for more flexible usage patterns, where the inferer policy can be
replaced in a live application without losing any state kept in
wrappers.

This change is currently non-breaking and is implemented separately
from the old wrapper system.

## [0.8.0] - 2025-05-28
- Added a new `RecurrentTracker` wrapper to handle recurrent
  inputs/outputs if the recurrent data is only needed durign network
  evaluation, f.ex. LSTM hidden states.
- `Inferer::infer_raw` now takes `&mut ScratchPadView` instead of an owned value, so wrappers can capture/mutate output data.

## [0.7.1] - 2025-05-19
- Support JSON output for benchmark command

## [0.7.0] - 2024-10-31
- Upgrade to compatibility with tract 0.21.7 and above.

## [0.6.1] - 2024-10-31
- Upgrade `time` to solve incompatibility with Rust 1.80.0
- Add upper bound for tract at 0.21.6 due to breaking upstream change

## [0.6.0] - 2024-02-12
- Upgrade to `perchance` v0.5
- Upgrade `tract` to 0.21.0

## [0.5.1] - 2024-01-18
- Do not ignore output shapes when constructing `onnx`-based inference models.

## [0.5.0] - 2023-10-09

- Mark cervo_runtime::BrainId as #[must_use]
- Move the CLI tool to a separate crate `cervo-cli`. The installed name is unchanged. This avoids some dependencies.
- Upgrade tract to 0.20.0

## [0.4.0] - 2022-11-23

- Upgrade to `perchance` v0.4.0, which removes dependency on `macaw` and `glam`

## [0.3.0] - 2022-10-04

### Batching

This update focuses on improving performance and simplifying
batching. To do this; there's a few changes that'll require updating
your code. These should generally be fairly simple, and in some-cases
as simple as removing `.to_owned()` and renaming a function call.

- There's no owned strings in inputs and outputs. This saves
  (`model-input-arity + model-output-arity) * avg-batch-size`
  string allocations on each call.
  - The values are still owned, as they act as sink/sourced. However;
 internally there's been multiple allocations removed per batch
 element.
- The `InfererExt::infer` trait has been deprecated in favor of
  `InfererExt::infer_single` and `InfererExt::infer_batch` which
  allows some small optimizations while clarifying the API.

Thus, to upgrade you'll need to:

- Change how you build your batches to cache the input names.
- Update which infer call function you use.

There's also a new `Batcher` which will help with batch building,
while also improving performance. Using this helper will further
reduce the number of allocations per call. On average, this is about a
10% performance gain. The easiest way to use this is using the
`.into_batched()` on your existing inferer.

Other changes:

- To match the new `Inferer` API; `NoiseGenerators` have to generate
  noise in-place instead of into a new vector.

### Cervo Runtime

As part of the performance work it's also become easier to deal with
multiple models using the new Cervo Runtime. The runtime helps with
managing multiple inference engines and their execution time,
enforcing batching and simplifying ownership. While someone already
running with a batched mode will not see huge gains, it should
hopefully provide a good building block and make adaptation easier in
an ECS.

### Other changes

- In general, all forward passes of an inferer should now be
  immutable, simplifying synchronization.

## [0.2.0] - 2022-07-11

- Upgrade all dependencies to tract 0.17.1
  - Fixes a memory leak when repeatedly creating inferers

### NNEF

- The initialization routine is now global and called `init`, instead of per-thread.

## [0.1.0] - 2022-06-02

Initial release.

<!-- next-url -->
[Unreleased]: https://github.com/EmbarkStudios/cervo/compare/0.9.2...HEAD
[0.9.2]: https://github.com/EmbarkStudios/cervo/compare/0.9.1...0.9.2
[0.9.1]: https://github.com/EmbarkStudios/cervo/compare/0.9.0...0.9.1
[0.9.0]: https://github.com/EmbarkStudios/cervo/compare/0.8.0...0.9.0
[0.8.0]: https://github.com/EmbarkStudios/cervo/compare/0.7.1...0.8.0
[0.7.1]: https://github.com/EmbarkStudios/cervo/compare/0.7.0...0.7.1
[0.7.0]: https://github.com/EmbarkStudios/cervo/compare/0.6.1...0.7.0
[0.6.1]: https://github.com/EmbarkStudios/cervo/compare/0.6.0...0.6.1
[0.6.0]: https://github.com/EmbarkStudios/cervo/compare/0.5.1...0.6.0
[0.5.1]: https://github.com/EmbarkStudios/cervo/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/EmbarkStudios/cervo/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/EmbarkStudios/cervo/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/EmbarkStudios/cervo/compare/0.2.0...0.3.0
[0.2.0]: https://github.com/EmbarkStudios/cervo/compare/0.1.1...0.2.0
[0.1.0]: https://github.com/EmbarkStudios/cervo/releases/tag/0.1.0
