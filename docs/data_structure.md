## Dataset structure

Note: file extensions may vary.

### Case 4: category sub-directories, each containing `LQ/` and `GT/`

Each category sub-directory must contain a `LQ/` sub-directory (low-quality / degraded images)
and a `GT/` sub-directory (ground-truth / clean images). Files are matched by name.

```
  data/dataset1
  |-- Blur
  |   |-- LQ
  |   |   |-- xxx.png
  |   |   `-- yyy.png
  |   `-- GT
  |       |-- xxx.png
  |       `-- yyy.png
  `-- Haze
      |-- LQ
      |   |-- xxx.png
      |   `-- yyy.png
      `-- GT
          |-- xxx.png
          `-- yyy.png
```

All pairs across all categories are collected, shuffled, and split automatically into
`train` / `val` subsets according to `val_ratio` (default: 10 % for validation).