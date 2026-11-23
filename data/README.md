# Local market-data snapshot

The notebook writes `adjusted_open.csv`, `adjusted_close.csv`, `treasury_yield.csv` and `manifest.json` to `data/` under its working directory. When run from the notebook folder, this is `notebooks/data/`. SHA-256 hashes identify the three input files. Section 2 of the executed notebook records the hashes used for the reported results.

These files remain on the local machine. The first run in a new location requires an internet connection to Yahoo Finance. To reproduce the same prices offline, copy all four snapshot files together into that location's `data/` folder.
