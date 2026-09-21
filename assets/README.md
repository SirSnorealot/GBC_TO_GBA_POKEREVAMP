# assets/

Static, committed configuration for asset acquisition.

- `manifests/sources.json` — upstream repository URLs, discovery globs, exclusion rules,
  the name-normalisation dictionary, and the curated sample lists.

No game graphics are stored here. Running `gbc_to_gba_pokerevamp assets bootstrap`
(or `scripts/bootstrap_assets.py`) shallow-clones the pret decompilation repositories
(pokered, pokeyellow, pokegold, pokecrystal, pokefirered, pokeemerald) into the gitignored
`vendor/` directory and copies only front battle sprites into the gitignored
`data/references/` tree, prefixed by game (`rb_`, `yellow_`, `gold_`, `silver_`, `crystal_`,
`frlg_`, `emerald_`). Generated manifests with per-file attribution and commit SHAs are
written to `data/manifests/`:

- `data/manifests/sources.json` — one record per copied image (source repo, commit, path, size, colour count)
- `data/manifests/pokemon_pairs.json` — Crystal -> FRLG/Emerald pairs by canonical species name
- `data/manifests/trainer_pairs.json` — high-confidence trainer class/character pairs
- `data/sample_set.json` — curated starter subset used for tuning contact sheets
