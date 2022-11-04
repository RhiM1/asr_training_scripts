import audioop
from lib2to3.pgen2 import token
import pickle as pkl
import numpy as np
from typing import List
import torch
import os
# from tools import load_corpus
import tools
import time
from model_utils import load_checkpoint, load_nemo_checkpoint, load_model, load_sc_model, write_to_log, decode_lm
import argparse
from ami_ex_dataset import ami_ex_dataset
from non_iid_ex_dataloader import get_eval_ex_dataloader, get_ex_data_loader

def get_logits(filename):
    with open(filename, 'rb') as f:
        data = pkl.load(f)

    return data

def pad_np_array_to_fixed(np_array_list):
   
    newLen = max([len(x) for x in np_array_list])
    print("Max array length:", newLen)
    # padded_array = torch.empty(0, dtype=float)

    for i in range(len(np_array_list)):
        # if i < 10:
            # print("Initial shape:", np.shape(np_array_list[i]))
        np_array_list[i] = np.pad(np_array_list[i], ((0, newLen - np.shape(np_array_list[i])[0]), (0, 0)), mode = 'empty')
        # if i < 10:
            # print("Padded shape:", np.shape(np_array_list[i]))
    
    padded_array = np.stack(np_array_list, axis = 0)
        
    return padded_array

def main(args):

    model = load_model(args) if args.self_conditioned == False else load_sc_model(args)
    tokenizer = model.tokenizer

    

    # with torch.no_grad:
    split = "test"
    # labelsSave = "./labels/" + split + "_4xCTC.pt"
    # labelLensSave = "./labels/" + split + "_lens_4xCTC.pt"
    shuffle = True
    
    # ami_dict = load_corpus(target_folder="/home/acp20rm/data/ami/ami_dataset/", prefix_path="/home/acp20rm/data/ami/")
    # ami_dict = load_feats_corpus(target_folder="/store/store1/data/ami/ami_dataset/", prefix_path="/store/store1/data/ami/")
    ami_dict = tools.load_corpus()
    # corpus = ami_dict[split]

    # print(corpus)


    strLabelsFile = "/home/acp20rm/data/ami/labels/sc-ctc-list/" + split + "_labels.pt"
    strLabelsLensFile = "/home/acp20rm/data/ami/labels/sc-ctc-list/" + split + "_labels_lens.pt"

    labels = torch.load(strLabelsFile)
    labels_lens = torch.load(strLabelsLensFile)

    dataloader = get_eval_ex_dataloader(
        ami_dict[split], 
        labels,
        labels_lens,
        max_duration=args.max_duration, 
        return_speaker=True, 
        batch_size=args.num_meetings, 
        concat_samples=args.concat_samples,
        split_speakers=args.split_speakers,
        gap=args.gap,
        speaker_gap=args.speaker_gap,
        single_speaker_with_gaps=args.single_speaker_with_gaps,
    )


    for batchID, batch in enumerate(dataloader):
        if batchID < 1:
            # print("features size:", batch['features'].size())
            # print("features_lens size:", batch['features_lens'].size())
            # print("labels size:", batch['labels'].size())
            # print("labels_lens size:", batch['labels_lens'].size())
            # # print("tokens size:", batch['tokens'].size())
            # # print("tokens_lens size:", batch['tokens_lens'].size())
            # print("labels:", batch['labels'])
            # print("text length:", len(batch['texts']))
            print("len batch:", len(batch['labels_lens']))
            for i in range(len(batch['labels_lens'])):
                print(i, batch['labels'][i], batch['labels_lens'][i], batch['audio_lens'][i])
                # print(i, batch['features_lens'][i])
                # print(i, batch['labels_lens'][i])
                # print(i, batch['texts'][i])
            # quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='/exp/exp1/acp21rjf/deliberation/Custom/model_configs/Hconformer_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer', type=str, default='./tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--max_duration', type=float, default=60, help='max duration of audio in seconds')
    parser.add_argument('--num_meetings', type=int, default=1, help='number of meetings per batch')

    # parser.add_argument('--batch_size', type=int, default=10)
    # parser.add_argument('-save_logits', '--save_logits_location', default='', help='path to save logits')   
    # parser.add_argument('-load_logits', '--load_logits_location', default='', help='path to load logits')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='../checkpoints_done/longcontextscctc/')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_61_id_9.pt')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--sweep', action='store_true', help='run wandb search for language model weight')
    parser.add_argument('-sc','--self_conditioned', action='store_false', help='use self-conditioned model (STORE FALSE)')
    # parser.add_argument('--beam_prune_logp', type=float, default=-10.0)
    # parser.add_argument('--token_min_logp', type=float, default=-5.0)
    # parser.add_argument('--save_text', action='store_true', help='save text output')

    parser.add_argument('--config_from_checkpoint_dir', action='store_false', help='load config from checkpoint dir') ##chane
    parser.add_argument('-cer', '--cer', action='store_true', help='compute CER instead of WER')

    # parser.add_argument('--ctx_model', action='store_true', help='use context model')

    # parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')


    parser.add_argument('--return_attention', action='store_true', help='return attention')
    parser.add_argument('--save_attention', action='store_true', help='save attention')
    parser.add_argument('--save_attn_dir', type=str, default='./attns')

    parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    parser.add_argument('--concat_samples', action='store_true', help='if set, will concat cuts from same meeting instead of stacking them')
    parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')
    parser.add_argument('-dnpsl','--do_not_pass_segment_lens', action='store_true', help='if set, will not pass segment lens to the model, used with concat_samples for single segment models')

    parser.add_argument('-sclite','--sclite', action='store_true', help='if set, will eval with sclite')
    parser.add_argument('-sclite_dir', '--sclite_dir', type=str, default='./trns')

    parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')
    parser.add_argument('-save_labels', default='', type=str, help='save predicted frame labels for use as exemplars')

   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    if args.config_from_checkpoint_dir == True:
        dir_contents = os.listdir(args.checkpoint_dir)
        config = [el for el in dir_contents if el.endswith('.yaml')]
        assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
        args.model_config = os.path.join(args.checkpoint_dir, config[0])
        print(f'Loading config from checkpoint dir: {args.model_config}')


    # if args.load_logits_location != '' and os.path.exists(args.load_logits_location) == False:
    #     raise ValueError(f'{args.load_logits_location} does not exist')

    # if args.save_text == True and os.path.exists('./txt_outputs') == False:
    #     os.mkdir('./txt_outputs')
  
    main(args)