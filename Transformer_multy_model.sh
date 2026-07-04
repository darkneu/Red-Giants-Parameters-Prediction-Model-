nohup $(which python) multy_model_v5.py \
    --data /Users/yzc/redgaints/ml_data_v2/train.csv \
    --target_cols "age" \
    --stellar_cols "numax_norm,dnu_norm,teff_norm,feh_linear" \
    --log_stellar_cols "numax_norm,dnu_norm" \
    --seq_norm asinh_robust \
    --seq_clip_sigma 0 \
    --max_seq_len 6000 \
    --patch_size 20 \
    \
    --base_ch 32 \
    --dropout 0.35 \
    --use_transformer \
    --n_trans_layers 3 \
    --n_heads 4 \
    --stellar_hidden 128 \
    --fusion_mode mlp_pred \
    --fusion_hidden 96 \
    \
    --batch_size 32 \
    --epochs_spec 250 \
    --epochs_stellar 150 \
    --epochs_fusion 100 \
    --lr_spec 3e-4 \
    --lr_stellar 5e-4 \
    --lr_fusion 3e-4 \
    --weight_decay 0.05 \
    --patience 30 \
    --val_ratio 0.15 \
    --grad_clip 0.5 \
    \
    --huber_delta 1.0 \
    --target_weight_momentum 0.9 \
    --target_weight_min 0.5 \
    --target_weight_max 3.0 \
    \
    --use_amp \
    --use_ema \
    --ema_decay 0.999 \
    --seed 42 \
    > multy_model_v5.log 2>&1 &