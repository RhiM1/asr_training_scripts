echo '<<< TRAINING SMALL SC-CTC MODEL >>>'

# python train_H.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/sc_ctc_lr1e-6_wd1e-2/' \
#     --model_config '../model_configs/conformer_sc_ctc_small.yaml' \
#     --min_lr 1e-6 \
#     --max_lr 3e-5 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-2 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/sc_ctc_lr1e-7_wd1e-4/' \
#     --model_config '../model_configs/conformer_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-4 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/sc_ctc_lr1e-7_wd1e-3/' \
#     --model_config '../model_configs/conformer_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-3 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/sc_ctc_lr1e-7_wd1e-2/' \
#     --model_config '../model_configs/conformer_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-2 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1


python train_H_ex.py --checkpoint '' \
    --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-6_wd1e-4/' \
    --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
    --min_lr 1e-6 \
    --max_lr 3e-5 \
    --step_size 150 \
    --step_size 150 \
    --weight_decay 1e-4 \
    --accumulate_gradients 1 \
    --clip_gradients \
    --clip_gradients_value 10 \
    --micro_batch_duration 180 \
    --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
    --do_not_pass_segment_lens \
    --wandb_id '' \
    --wandb_project 'ex-TIMIT' \
    --epochs 1

# python train_H_ex.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-6_wd1e-3/' \
#     --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
#     --min_lr 1e-6 \
#     --max_lr 3e-5 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-3 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H_ex.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-6_wd1e-2/' \
#     --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
#     --min_lr 1e-6 \
#     --max_lr 3e-5 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-2 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H_ex.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-7_wd1e-4/' \
#     --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-4 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H_ex.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-7_wd1e-3/' \
#     --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-3 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

# python train_H_ex.py --checkpoint '' \
#     --checkpoint_dir './checkpoints/ex_sc_ctc_lr1e-7_wd1e-2/' \
#     --model_config '../model_configs/conformer_ex_sc_ctc_small.yaml' \
#     --min_lr 1e-7 \
#     --max_lr 3e-6 \
#     --step_size 150 \
#     --step_size 150 \
#     --weight_decay 1e-2 \
#     --accumulate_gradients 1 \
#     --clip_gradients \
#     --clip_gradients_value 10 \
#     --micro_batch_duration 180 \
#     --schedular_data 'sc-ctc_ami_baseline_scheduler.json' \
#     --do_not_pass_segment_lens \
#     --wandb_id '' \
#     --wandb_project 'ex-TIMIT' \
#     --epochs 1

echo '<<< WE ARE DONE! >>>'


# --micro_batch_number 0 \

# micro_batch_number = batch size 
# unless micro_batch_duration is > 0 then utterances from the same discourse are passed together up to a max duration
# step_size = step size up, step size down is step_size*4 
