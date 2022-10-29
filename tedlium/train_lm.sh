echo '<<< TRAINING AUTOREGRESSIVE TLM >>>'

python train_LM.py --checkpoint '' \
    --checkpoint_dir './checkpoints_15_test_s4' \
    --model_config './lm/decoder_test.yaml' \
    --min_lr 1e-6 \
    --max_lr 2.75e-4 \
    --step_size 50 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 60 \
    --micro_batch_number 30 \
    --max_allowed_utterance_gap 3.0 \
    --wandb_id '' \
    --save_top_k 1 \
    --schedular_data '15_tests4.json' 
    



#GCC/9.3.0
#    --wandb_id '3bc1yvgb' \

#checkpoint_26_id_78.pt

# step_size = step size up, step size down is step_size*4 
