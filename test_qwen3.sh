CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters best-model \
    --val_dataset ~/test_binary_65535_1200.jsonl \
    --merge_lora true \
    --tensor_parallel_size 16 \
    --stream true \
    --infer_backend pt 