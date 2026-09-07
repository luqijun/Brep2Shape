# 2. 训练（无预训练模型，从头自监督）
DATASET_DIR=data/mechcad_processed BATCH_SIZE=16 BASE_BATCH_SIZE=8 EPOCHS=50 bash scripts/run_step_pretrain.sh