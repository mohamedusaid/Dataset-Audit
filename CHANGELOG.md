# Changelog

## 0.3.0

- Stitch documents across consecutive shard boundaries within a split. Document,
  duplicate, and leakage counts can therefore differ from 0.2.0 by the number of
  affected shard boundaries.
- Add duplicate-excess token metrics and tighten quality/split/integrity handling.

## 0.2.0

- Add validated configuration, secure archive extraction, deterministic sampling,
  atomic artifacts, packaging, CI, and expanded test coverage.
