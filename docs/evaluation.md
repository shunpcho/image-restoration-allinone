# Evaluation
The evaluation consists of the comparison of the restored/generated images with the reference ground truth images. Please ensure all images are processed using a unified model to maintain consistency across different test scenarios.

## Calculate Score

**Final_Score = PSNR (Y) + 10 * SSIM (Y) - 5 * LPIPS**

## Save Outpu Images

```python
from PIL import Image
img = Image.open(img_path)
if img.mode != "RGB":
    img = img.convert("RGB")
save_kwargs = dict(format="JPEG", quality=96, subsampling=0, optimize=True)
img.save(out_path, **save_kwargs)
```

## Final Score
| Rank | Final Score | PSNR | SSIM | LPIPS |
| -- | -- | -- | -- | -- |
| 1 | 33.86 | 27.49 | 0.81 | 0.34 |
|2|33.58|26.71|0.82|0.26|
|3|32.63|26.05|0.8|0.29|
|4|32.62|26.17|0.79|0.3|
|5|32.61|26.08|0.8|0.29|