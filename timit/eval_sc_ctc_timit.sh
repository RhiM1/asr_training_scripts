echo '<<< EVALUATING SMALL SC-CTC MODEL >>>'
python eval_ctc_notBPE.py --max_duration 90 \
    -dnpsl \
    --checkpoint_dir './checkpoints/sc-ctc/' \
    --checkpoint 'checkpoint_38_id_18.pt' \
    -lm './lm/3gram-6mix.arpa' \
    --alpha 0.5 \
    --split 'TRAIN' \
    # -save_labels '/home/acp20rm/data/ami/labels/sc-ctc-list/'

echo '<<< WE ARE DONE! >>>'


# --num_meetings 3 \

# micro_batch_number = batch size 
# step_size = step size up, step size down is step_size*4 
