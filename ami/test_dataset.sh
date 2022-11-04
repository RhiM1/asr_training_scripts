CUDA_VISIBLE_DEVICES=0 python testDataset.py  \
    --num_meetings 3 \
    -dnpsl \
    --checkpoint_dir './checkpoints/sc-ctc/' \
    --checkpoint 'checkpoint_182_id_5.pt' \
    -lm './lm/3gram-6mix.arpa' \
    --alpha 0.5 \
    --split 'test' \
    -save_labels '/home/acp20rm/data/ami/labels/sc-ctc-list/'


    # --checkpoint_dir './checkpoints/sc-ctc/' \
    # --checkpoint '' \
    # --model_config '../model_configs/conformer_sc_ctc_bpe_small.yaml' \
    # --batch_size 1 \
    # --save_logits_location '' \
    # -lm './3gram-6mix.arpa' \
    # --beam_size 150 \
    # --beam_prune_logp -10.466836794439956 \
    # --token_min_logp -4.178187657336318 \
    # --split 'test' \
    # --alpha 0.5 \
    # --beta 0.8 \
    # -sc
    # -save_labels '/home/acp20rm/data/ami/labels/sc-ctc-list/'

    #./logits/logits.txt