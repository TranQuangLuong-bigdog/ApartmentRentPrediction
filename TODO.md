# TODO - Registry Enhancement (PR6)

- [ ] Update `src/model_registry/model_metadata.py`:
  - Ensure schema includes `dataset_hash`, `feature_schema`, `categorical_columns`, `numerical_columns`, `pipeline_version`, `preprocessing_version`, `active_model` (via registry top-level pointer).
  - Add defaults for backward compatibility with old registry entries.

- [ ] Update `src/model_registry/registry.py`:
  - Add file lock protecting `trained_models/registry.json`.
  - Implement new methods without changing current public API:
    - `check_compatibility()`
    - `compare_models()`
    - `search_model()`
    - `export_registry()`
    - `upgrade_registry()`
    - `migrate_registry()`
  - Ensure top-level fields exist: `active_model`, `latest_model`, `best_model`.

- [ ] Add unit tests `tests/test_registry.py`:
  - Validate migrate/upgrade behavior from current `trained_models/registry.json`.
  - Validate `check_compatibility` pass/fail.
  - Validate `export_registry` outputs expected files.
  - Validate search/compare return deterministic order.

- [ ] Run tests: `pytest -q`.

