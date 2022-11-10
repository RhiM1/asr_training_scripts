echo '<<< EVALUATING SMALL SC-CTC MODEL >>>'
CUDA_VISIBLE_DEVICES=0 python eval_ex_ctc_h_init.py --max_duration 0 \
    --num_meetings 3 \
    -dnpsl \
    --checkpoint_dir './checkpoints/ex-V/' \
    --checkpoint 'checkpoint_5_id_27.pt' \
    -lm './lm/3gram-6mix.arpa' \
    --alpha 0.5 \
    --split 'dev' \

echo '<<< WE ARE DONE! >>>'


# --ex_model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
# --ex_checkpoint_dir './checkpoints/sc-ctc/' \
# --ex_checkpoint 'checkpoint_209_id_5.pt' \

# -save_labels '/home/acp20rm/data/ami/labels/sc-ctc-list/'
# micro_batch_number = batch size 
# unless micro_batch_duration is > 0 then utterances from the same discourse are passed together up to a max duration
# step_size = step size up, step size down is step_size*4 
