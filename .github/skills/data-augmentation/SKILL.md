---
name: data-augmentation
description: >
  Image Restoration モデルのデータ拡張パイプラインを構築する。
  albumentations を使用して degraded / clean ペア画像に同一の空間変換を適用する。
  データ拡張、augmentation pipeline、pair-consistent transform、albumentations の
  実装・設定を行う場合に使用する。
---

# Skill: Data Augmentation

## Overview

Image Restoration モデルの汎化性能を高めるためのデータ拡張スキルです。
Restoration タスクでは **劣化画像（degraded）とクリーン画像（clean）に必ず同一の空間変換を適用** する必要があります。
`albumentations` ライブラリを主軸に、ペア整合が保証された拡張パイプラインを構築します。

## Applicable Agents

- CV Data Agent（拡張パイプラインの組み込み）

## Library Choice

`albumentations` を第一選択とします。

- 高速かつ豊富な変換を提供
- `additional_targets` を使って degraded / clean の 2 枚に **同一の空間変換** を適用できる
- ランダムシードを介した再現可能な拡張をサポート

## Pair-Consistent Augmentation

Image Restoration では degraded と clean の空間的対応を壊さないことが必須です。
`albumentations` の `additional_targets` を使用して両画像に同一変換を適用します。

```python
import albumentations as albu
from albumentations.pytorch import ToTensorV2

transform = albu.Compose(
    [
        albu.RandomCrop(height=128, width=128),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        ToTensorV2(),
    ],
    additional_targets={"clean": "image"},  # "degraded" が image, "clean" が追加ターゲット
)

# 使用例
result = transform(image=degraded_np, clean=clean_np)
degraded_tensor = result["image"]
clean_tensor = result["clean"]
```

## Augmentation Categories

### 空間変換（Spatial Transforms）

degraded / clean の両画像に **同一変換** を適用します。

| 変換 | `albumentations` クラス | Restoration での利用 |
|------|------------------------|---------------------|
| ランダムクロップ | `RandomCrop` | 学習時のパッチ抽出（必須） |
| 水平反転 | `HorizontalFlip` | 全タスク共通 |
| 垂直反転 | `VerticalFlip` | 全タスク共通 |
| 90° 回転 | `RandomRotate90` | 全タスク共通 |
| 任意角度回転 | `Rotate` | 超解像・ノイズ除去（境界処理に注意） |

### ピクセル変換（Pixel Transforms）

**clean 画像にのみ適用してはならない** 変換です。
Restoration タスクでは degraded / clean の輝度・色調整を **同時かつ同一** に行うか、あるいはまったく適用しないかを選択します。

| 変換 | 推奨適用方針 |
|------|------------|
| 明度・コントラスト調整 | 両画像に同一パラメータで適用（`additional_targets` 利用）|
| 色相・彩度調整 | 両画像に同一パラメータで適用（`additional_targets` 利用）|
| ガウシアンノイズ付加 | **degraded のみ** に適用（合成劣化として `degradation.py` で管理）|
| ブラー | **degraded のみ** に適用（合成劣化として `degradation.py` で管理）|

> **注意**: ノイズ・ブラーなどの劣化付加はこの `SKILL.md` ではなく `cv-data-agent.agent.md` の `degradation.py` で管理します。データ拡張と劣化合成を明確に分離してください。

## Task-Specific Pipeline Examples

### 共通（ノイズ除去・デブラー・超解像）

```python
import albumentations as albu
from albumentations.pytorch import ToTensorV2

train_transform = albu.Compose(
    [
        albu.RandomCrop(height=128, width=128),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ],
    additional_targets={"clean": "image"},
)

val_transform = albu.Compose(
    [
        albu.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ],
    additional_targets={"clean": "image"},
)
```

### 超解像（LR / HR ペア）

```python
# LR のクロップサイズ = HR のクロップサイズ // scale
# LR と HR は解像度が異なるため、HR に対して RandomCrop を適用し
# LR には対応する領域を計算してクロップする（別途実装）
# albumentations の RandomCrop は同一サイズ前提のため、
# 超解像では HR をクロップしてから bicubic でダウンサンプリングするパイプラインを推奨する
```

## Implementation Guidelines

- 拡張パイプラインはトレーニング時のみ適用する。検証・テスト時は正規化のみ。
- `additional_targets` を使って degraded / clean の空間変換の一貫性を保証する。
- 劣化付加（ノイズ・ブラー等）は `degradation.py` で管理し、このパイプラインに含めない。
- 再現性のために `albumentations` の設定を `albu.to_dict(...)` / `albu.from_dict(...)` でシリアライズ可能にする。
- テスト時は全解像度（full-size）の画像を使用する。

## Common Pitfalls

- `additional_targets` を設定せずに変換を適用すると、degraded と clean の空間変換がずれる。
- 超解像タスクでは LR と HR の解像度が異なるため、`RandomCrop` を LR / HR 個別に計算する必要がある。
- 正規化は `albumentations.Normalize` でパイプライン内で行い、手動計算と混在させない。
- 検証・テスト時にランダム拡張（HorizontalFlip 等）を適用しないよう、train / val 用パイプラインを明確に分離する。
