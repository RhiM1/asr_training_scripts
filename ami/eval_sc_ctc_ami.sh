echo '<<< EVALUATING SMALL SC-CTC MODEL >>>'
python eval_ctc_h_init.py --max_duration 0 \
    --num_meetings 3 \
    -dnpsl \
    --checkpoint_dir './checkpoints/sc-ctc/' \
    --checkpoint 'checkpoint_153_id_95.pt' \
    -lm './lm/3gram-6mix.arpa' \
    --alpha 0.5 \
    --split 'test' \
    # -save_labels '/home/acp20rm/data/ami/labels/sc-ctc-list/'

echo '<<< WE ARE DONE! >>>'



# micro_batch_number = batch size 
# unless micro_batch_duration is > 0 then utterances from the same discourse are passed together up to a max duration
# step_size = step size up, step size down is step_size*4 
