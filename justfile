train_sample:
    train \
        --data-root data/samples \
        --epochs 100 \
        --val-interval 5 \
        --checkpoint-freq 50

train:
    train \
        --data-root data/train \
        --epochs 10 \
        --val-interval 2 \
        --checkpoint-freq 5 \
        --batch-size 16 \

mlflow:
    mlflow ui --port 5000
