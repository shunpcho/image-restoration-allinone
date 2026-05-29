---
name: image-deblurring
description: >
  画像デブラー（Image Deblurring）タスクを実装する。
  ブラーを含む劣化画像からシャープなクリーン画像を復元する教師あり学習モデルを構築する。
  モーションブラー、デフォーカスブラー、ガウシアンブラーに対応。
  NAFNet、MPRNet、DeblurGAN-v2 などのアーキテクチャを使用する場合に利用する。
---

# Skill: Image Deblurring

## Overview

画像デブラー（Image Deblurring）タスクを実装するためのスキルです。
教師あり学習により、ブラーを含む劣化画像（blurry）からシャープなクリーン画像（sharp）を復元します。
モーションブラーおよびデフォーカスブラーの両方をカバーします。

## Applicable Agents

- CV Data Agent（ブラー合成・ペアデータセット構築）
- CV Training Agent（モデルトレーニング）
- CV Evaluation Agent（PSNR / SSIM / LPIPS 評価）

## Task Definition

- **入力**: ブラー画像 (`torch.Tensor`, shape `[B, C, H, W]`, 値域 `[0, 1]`)
- **出力**: シャープ画像 (`torch.Tensor`, shape `[B, C, H, W]`, 値域 `[0, 1]`)
- **損失関数**: `L1Loss` + Perceptual Loss（VGG）+ `λ * FFT Loss`（周波数ドメイン損失）

## Blur Types

| ブラー種別      | 説明                     | 合成方法                                    |
| --------------- | ------------------------ | ------------------------------------------- |
| Gaussian blur   | ピントのズレを模擬       | `albumentations.GaussianBlur`               |
| Motion blur     | 手ブレ・被写体ブレを模擬 | ランダム方向・長さのカーネルで畳み込み      |
| Defocus blur    | レンズのデフォーカス     | 円形カーネル（disk kernel）で畳み込み       |
| Real-world blur | 実際のカメラ手ブレ       | GoPro / HIDE などの実劣化データセットを使用 |

## Recommended Architectures

| ユースケース     | 推奨アーキテクチャ               |
| ---------------- | -------------------------------- |
| 高品質・汎用     | `NAFNet`, `Restormer`, `MPRNet`  |
| Motion blur 特化 | `DeblurGAN-v2`, `MIMO-UNet`      |
| 軽量             | `GRL-B`（軽量版）、`FFDNet` 変形 |

## Benchmark Datasets

| データセット            | ブラー種別       | 備考                             |
| ----------------------- | ---------------- | -------------------------------- |
| GoPro Large             | Real motion blur | 2013 枚の train / 1111 枚の test |
| HIDE                    | Real motion blur | 人物シーンに特化                 |
| RealBlur-J / RealBlur-R | Real blur        | JPEG 版と RAW 版                 |
| DPDD                    | Defocus blur     | Dual-pixel データ付き            |

## Key Metrics

- **PSNR**：GoPro での NAFNet は約 33 dB
- **SSIM**
- **LPIPS**：知覚的シャープネスの評価に重要

## Implementation Checklist

- [ ] 合成ブラーカーネルの種類・サイズをハイパーパラメータとして外部設定できるようにする
- [ ] 学習時はランダムパッチ（例: 256×256）を使用し、推論時は full-size 画像を使用する
- [ ] `L1Loss` + Perceptual Loss を組み合わせた損失関数を実装する
- [ ] FFT Loss（周波数ドメイン L1）を追加することでシャープネスを向上させる（オプション）
- [ ] `torchmetrics.image.PeakSignalNoiseRatio` で PSNR を算出する
- [ ] `torchmetrics.image.StructuralSimilarityIndexMeasure` で SSIM を算出する
- [ ] `mlflow.log_params` でブラー設定・モデル設定を記録する
- [ ] `mlflow.log_metrics(step=iteration)` で validation ごとの loss / PSNR / SSIM を記録する
- [ ] `mlflow.pytorch.log_model` でトレーニング済みモデルを保存する

## Common Pitfalls

- 合成ブラーと実劣化データ（GoPro 等）でドメインギャップが生じる場合がある。実劣化データセットでの評価を必ず行う。
- Motion blur 合成時はブラーカーネルを画像ごとにランダムに変化させることでモデルの汎化性を高める。
- 周波数ドメイン損失（FFT Loss）は高周波成分の復元に有効だが、過度に適用するとリンギングアーティファクトが発生する場合がある。
- バッチ正規化は Restoration タスクでは不安定になることがあるため、Layer Normalization を優先する。
