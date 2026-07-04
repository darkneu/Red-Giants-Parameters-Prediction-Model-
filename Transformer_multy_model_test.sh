python multy_model_v5_test.py \
    --data /Users/yzc/redgaints/ml_data_v2/test.csv \
    --ckpt ./checkpoints/best_fusion.pt \
    --spectrum_prefix spec_ \
    --mode fusion \
    --id_col kic \
    --output ./checkpoints/age_predictions_fusion.csv \
    --save_metrics metrics.json