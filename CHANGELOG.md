# Changelog

## v0.9.7 (2026-11-04)

- **Native Python `rtools` engine** (`qradiomics.feature.rtools`): a port of the
  legacy radiomics-tools C++ `FeatureExtraction` binary (numpy + scipy;
  SimpleITK for image I/O only). Output matches the binary's printed values;
  neither ITK Python nor the compiled binary is required.
  - `qr extract --engine rtools` now uses the native port by default.
  - `rtools_backend="cxx"` keeps the legacy binary for parity checks.
  - `rtools_ibsi=True` selects an opt-in IBSI-1 profile
    (`rtools_ibsi_bin_width` / `rtools_ibsi_bin_number`), verified against the
    IBSI-1 digital phantom (`tests/fixtures/ibsi1_digital_phantom.json`).
- Tests: `tests/test_rtools_ibsi.py`, `tests/test_rtools_parity.py`.
