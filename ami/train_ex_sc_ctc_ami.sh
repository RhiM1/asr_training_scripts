echo '<<< TRAINING SMALL SC-CTC MODEL >>>'

CUDA_VISIBLE_DEVICES=0 python train_ex_H.py --checkpoint '' \
    --checkpoint_dir './checkpoints/ex-simple-eye/' \
    --model_config '../model_configs/conformer_ex_sc_ctc_bpe_small.yaml' \
    --ex_model_config '' \
    --ex_checkpoint_dir '' \
    --ex_checkpoint '' \
    --min_lr 1e-5 \
    --max_lr 3e-4 \
    --step_size 150 \
    --step_size 150 \
    --accumulate_gradients 5 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 0 \
    --micro_batch_number 36 \
    --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --wandb_project 'ex-AMI'

echo '<<< WE ARE DONE! >>>'


# --ex_model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
# --ex_checkpoint_dir './checkpoints/sc-ctc/' \
# --ex_checkpoint 'checkpoint_209_id_5.pt' \
# --output100


# micro_batch_number = batch size 
# unless micro_batch_duration is > 0 then utterances from the same discourse are passed together up to a max duration
# step_size = step size up, step size down is step_size*4 
