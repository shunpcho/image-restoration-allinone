train_sample:
    train \
        --config config_yaml/config.yaml \
        data.data_root data/samples \
        train.batch_size 4 \
        model.arch_name "Restormer" \
        model.restormer.dim 8

train:
    train \
        --data-root data/train \
        --epochs 10 \
        --val-interval 2 \
        --checkpoint-freq 5 \
        --batch-size 16 \

mlflow:
    mlflow ui --port 5000
