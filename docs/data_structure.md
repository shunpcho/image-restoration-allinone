## Dataset structure

Note: file extensions may vary.

### Case 1: paired images in a single directory (matched by filename keywords)

Keyword mapping:
- `_mean`: clean image
- `_real`: degraded image

```
  data/dataset1
  |-- xxx_mean.png
  |-- xxx_real.png
  |-- yyy_mean.png
  |-- yyy_real.png
  :
  :
  `-- zzz_real.png
```

### Case 2: `train`/`val` subdirectories and paired images in a single directory (matched by filename keywords)

Keyword mapping:
- `_mean`: clean image
- `_real`: degraded image

```
  data/dataset1
  |-- train
  |   |-- xxx_mean.png
  |   |-- xxx_real.png
  |   |-- yyy_mean.png
  |   `-- yyy_real.png
  `-- val
      |-- xxx_mean.png
      |-- xxx_real.png
      |-- yyy_mean.png
      `-- yyy_real.png
```

### Case 3: separate `clean`/`degre` directories

```
  data/dataset1
  |-- clean
  |   |-- xxx.png
  |   |-- yyy.png
  `-- degre
      |-- xxx.png
      `-- yyy.png
```