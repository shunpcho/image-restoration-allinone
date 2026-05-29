---
name: image-denoising
description: >
  Image Restoration タスクを実装する。
  劣化画像からクリーン画像を復元する教師あり学習モデルを構築する。
  AWGN、Poisson noise、salt-and-pepper、real-world noise に対応。
  DnCNN、NAFNet、Restormer などのアーキテクチャを使用する場合に利用する。
---

# Skill: Image Denoising

## Overview

Image Restoration タスクの一部を実装するスキルです。
教師あり学習により、劣化画像（degraded）からクリーン画像（clean）を復元します。

## Applicable Agents

- CV Data Agent（ノイズ付加・ペアデータセット構築）
- CV Training Agent（モデルトレーニング）
- CV Evaluation Agent（PSNR / SSIM / LPIPS 評価）

## Task Definition

- **入力**: 劣化画像 (`torch.Tensor`, shape `[B, C, H, W]`, 値域 `[0, 1]`)
- **出力**: 復元クリーン画像 (`torch.Tensor`, shape `[B, C, H, W]`, 値域 `[0, 1]`)
- **損失関数**: `L1Loss` または Charbonnier Loss（主流）、`+ λ * SSIMLoss`（任意）

## Recommended Architectures

| ユースケース                | 推奨アーキテクチャ    |
| --------------------------- | --------------------- |
| Blind / Non-blind AWGN 除去 | `DnCNN`, `FFDNet`     |
| 高品質・汎用                | `NAFNet`, `Restormer` |
| リアルタイム向け            | `DIDN`（軽量版）      |
| 医療画像                    | `U-Net` ベースの変形  |

## Benchmark Datasets

| データセット | 種別           | 用途                           |
| ------------ | -------------- | ------------------------------ |
| BSD68        | Synthetic AWGN | Gray image denoising 評価      |
| CBSD68       | Synthetic AWGN | Color image denoising 評価     |
| SIDD         | Real-world     | スマートフォンカメラノイズ除去 |
| DND          | Real-world     | デジタルカメラノイズ除去       |

## Key Metrics

- **PSNR** (Peak Signal-to-Noise Ratio)：値が高いほど良好（dB 単位）
- **SSIM** (Structural Similarity Index)：0〜1、高いほど良好
- **LPIPS**：値が低いほど知覚的品質が高い

## Implementation Checklist

- [ ] ノイズレベル σ（またはノイズレベルマップ）をハイパーパラメータとして外部設定できるようにする
- [ ] Blind denoising の場合、ノイズレベル推定を入力に含める（FFDNet 方式）
- [ ] 学習時はランダムパッチ（例: 128×128）を使用し、推論時は full-size 画像を使用する
- [ ] `L1Loss` または Charbonnier Loss を損失関数として使用する
- [ ] `torchmetrics.image.PeakSignalNoiseRatio` で PSNR を算出する
- [ ] `torchmetrics.image.StructuralSimilarityIndexMeasure` で SSIM を算出する
- [ ] `mlflow.log_params` でノイズレベル・モデル設定を記録する
- [ ] `mlflow.log_metrics(step=iteration)` で validation ごとの loss / PSNR / SSIM を記録する
- [ ] `mlflow.pytorch.log_model` でトレーニング済みモデルを保存する

## Common Pitfalls

- 入力テンソルの値域（`[0, 1]` または `[0, 255]`）をトレーニングと評価で統一する。
- Blind denoising モデルをノイズレベル固定で評価する場合、適切なノイズレベルマップを入力に与える。
- バッチ正規化は Restoration タスクでは不安定になることがあるため、Layer Normalization や Group Normalization を優先する。
