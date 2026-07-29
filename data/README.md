# Data

`sample/batch01_generation.zip` is a real generation archive included to test the repository's data
loader and integrity checker. It contains 600 trajectories (six agents across the first 100-task batch),
the frozen problem records, a health table, and a manifest.

SHA-256:

```text
19293c5b020b7ca9676431eda6fbce8574439d604f978f2c230195546c394795  batch01_generation.zip
```

The paper's complete 1,800 attempts and final repaired metrics are intentionally not committed here.
Place them in `artifacts/downloaded/` or supply their paths in the relevant notebook configuration cell.
The release should include their original ZIPs and SHA-256 values.
