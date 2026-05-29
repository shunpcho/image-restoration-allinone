---
name: super-resolution
description: >
  単一画像超解像（Single Image Super-Resolution, SISR）タスクを実装する。
  低解像度（LR）画像から高解像度（HR）画像を復元する教師あり学習モデルを構築する。
  スケール ×2 / ×3 / ×4 に対応。EDSR、SwinIR、Real-ESRGAN などのアーキテクチャを
  使用する場合に利用する。
---

# Skill: Super-Resolution

## Overview

単一画像超解像（Single Image Super-Resolution, SISR）タスクを実装するためのスキルです。
教師あり学習により、低解像度（LR）画像から高解像度（HR）画像を復元します。

## Applicable Agents

- CV Data Agent（HR→LR ダウンサンプリング・ペアデータセット構築）
- CV Training Agent（モデルトレーニング）
- CV Evaluation Agent（PSNR / SSIM / LPIPS 評価）

## Task Definition

- **入力**: 低解像度画像 LR (`torch.Tensor`, shape `[B, C, H, W]`, 値域 `[0, 1]`)
- **出力**: 高解像度画像 SR (`torch.Tensor`, shape `[B, C, H*scale, W*scale]`, 値域 `[0, 1]`)
- **スケール因子**: ×2 / ×3 / ×4 が一般的
- **損失関数**: `L1Loss`（Pixel loss）+ Perceptual Loss（VGG）+ GAN Loss（オプション）

## Degradation Models

| 劣化モデル                       | 説明                                     |
| -------------------------------- | ---------------------------------------- |
| Bicubic downsampling             | 古典的なベースライン（SRCNN 等で使用）   |
| Blur + Downscale + Noise (BD)    | 現実的な劣化の近似                       |
| Real-ESRGAN degradation pipeline | 複合的・ランダムな劣化（実写画像に対応） |

## Recommended Architectures

| ユースケース          | 推奨アーキテクチャ      |
| --------------------- | ----------------------- |
| 精度重視              | `SwinIR`, `HAT`, `DRLN` |
| 軽量・リアルタイム    | `IMDN`, `RFDN`, `ABPN`  |
| 知覚的品質重視（GAN） | `ESRGAN`, `Real-ESRGAN` |
| 汎用 baseline         | `EDSR`                  |

## Benchmark Datasets

| データセット                | 用途                                     |
| --------------------------- | ---------------------------------------- |
| DIV2K (800 train / 100 val) | 学習・検証の標準データセット             |
| Set5 / Set14 / BSD100       | 標準テストセット（bicubic ×2 / ×3 / ×4） |
| Urban100                    | 高頻度テクスチャ（建物・格子）の評価     |
| Manga109                    | マンガ・イラスト画像の評価               |

評価は **Y チャンネル（YCbCr 変換後）** で PSNR / SSIM を算出するのが標準。

## Key Metrics

- **PSNR** (Y チャンネル)：×4 EDSR で約 32 dB が目安
- **SSIM** (Y チャンネル)
- **LPIPS**：知覚的品質の評価（GAN ベースモデルで特に重要）

## Implementation Checklist

- [ ] スケール因子を `@dataclass(slots=True)` の設定フィールドとして定義する
- [ ] HR→LR の劣化パイプラインを `cv-data-agent` の `degradation.py` に実装する
- [ ] 学習時の LR パッチサイズは `HR_patch_size // scale` に合わせる
- [ ] `PixelShuffle`（sub-pixel convolution）または補間ベースの upsampler を使用する
- [ ] 評価前に画像を YCbCr 変換し、Y チャンネルで PSNR / SSIM を算出する
- [ ] `torchmetrics.image.PeakSignalNoiseRatio` で PSNR を算出する
- [ ] `torchmetrics.image.StructuralSimilarityIndexMeasure` で SSIM を算出する
- [ ] `mlflow.log_params` でスケール因子・劣化設定・モデル設定を記録する
- [ ] `mlflow.log_metrics(step=iteration)` で validation ごとの loss / PSNR / SSIM を記録する
- [ ] `mlflow.pytorch.log_model` でトレーニング済みモデルを保存する

## Common Pitfalls

- PSNR 評価時は画像境界のクロップ（通常 `scale` px）を行うのが標準。
- RGB / YCbCr の評価軸を論文と合わせる（ベンチマーク比較時に特に注意）。
- Bicubic ベースライン（`torchvision.transforms.functional.resize` with bicubic）と比較することで性能の基準を確認する。
- GAN ベースのモデルは知覚的品質は高いが PSNR / SSIM は低下する傾向があるため、目的に応じてモデルを選択する。
