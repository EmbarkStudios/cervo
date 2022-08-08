<!-- markdownlint-disable blanks-around-headings blanks-around-lists no-duplicate-heading -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- next-header -->
## [Unreleased] - ReleaseDate

This update focuses on improving performance and simplifying
batching. To do this; there's a few changes that'll require updating
your code. These should generally be fairly simple, and in some-cases
as simple as removing `.to_owned()` and renaming a function call.

* There's no owned strings in inputs and outputs. This saves
  (`model-input-arity + model-output-arity) * avg-batch-size`
  string allocations on each call.
  * The values are still owned, as they act as sink/sourced. However;
	internally there's been multiple allocations removed per batch
	element.
* The `InfererExt::infer` trait has been deprecated in favor of
  `InfererExt::infer_single` and `InfererExt::infer_batch` which
  allows some small optimizations while clarifying the API.

Thus, to upgrade you'll need to:

* Change how you build your batches to cache the input names.
* Update which infer call function you use.

There's also a new `Batcher` which will help with batch building,
while also improving performance. Using this helper will further
reduce the number of allocations per call. On average, this is about a
10% performance gain. The easiest way to use this is using the
`.into_batched()` on your existing inferer.

Other changes:

* To match the new `Inferer` API; `NoiseGenerators` have to generate
  noise in-place instead of into a new vector.

## [0.2.0] - 2022-07-11

* Upgrade all dependencies to tract 0.17.1
  * Fixes a memory leak when repeatedly creating inferers

### NNEF

* The initialization routine is now global and called `init`, instead of per-thread.

## [0.1.0] - 2022-06-02

Initial release.

<!-- next-url -->
[Unreleased]: https://github.com/EmbarkStudios/cervo/compare/0.2.0...HEAD
[0.2.0]: https://github.com/EmbarkStudios/cervo/compare/0.1.1...0.2.0
[0.1.1]: https://github.com/EmbarkStudios/cervo/compare/0.1.0...0.1.1
[0.1.0]: https://github.com/EmbarkStudios/cervo/releases/tag/0.1.0
