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
# from ami_ex_dataset import ami_ex_dataset
from non_iid_ex_dataloader import get_eval_ex_dataloader, get_ex_data_loader
from lhotse.dataset.collation import collate_audio
from lhotse.cut import CutSet
from lhotse.dataset.sampling.simple import SimpleCutSampler
from lhotse import SupervisionSegment

class BasicDataset(torch.utils.data.Dataset):

  def __getitem__(self, cuts: CutSet) -> dict:
    # cuts = cuts.sort_by_duration()
    print(cuts)
    rIDs = list(cuts.ids)

    for cut in cuts:
        rIDs.append(cut.supervisions[0].recording_id)
    audios, audio_lens = collate_audio(cuts)
    return {
        "recording_IDs": rIDs,
        "audio": audios,
        "audio_lens": audio_lens
    }


def main(args):
    split = "test"
    corpus = tools.load_corpus()[split]
    print(corpus)
    sample_rate = 16000

    if args.save_labels == '':
        save_labels = False
    else:
        save_labels = True
        strLabelFile = args.save_labels + args.split + "_labels.pt"
        strLabelsLensFile = args.save_labels + args.split + "_labels_lens.pt"
        labels = []
        # labels = torch.zeros((len(corpus), 1), dtype = torch.long)
        labels_lens = torch.zeros(len(corpus), dtype = torch.long)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = load_sc_model(args).to(device)
    model.eval()
    ds = BasicDataset()
    sampler = SimpleCutSampler(
        corpus,
        max_duration = 25
    )
    dl = torch.utils.data.DataLoader(ds, sampler = sampler)

    # tokenizer = m
    
    # model.to(device)
    i = 0
    # j = 0
    # max_len_samps = 0
    
    supervisions = []
    timeCorrection = 1


    for batchCuts in dl:

        print(batchSups)
        print(batchRecs)

        print("batchCuts", batchCuts)
        
        audios, audio_lengths = collate_audio(batchCuts)
        # audios = audios.reshape(-1, audios.shape[-1]).to(device)

        audios = audios.reshape(-1, audios.shape[-1]).to(device)
        audio_lengths = audio_lengths.reshape(-1).to(device)
        rIDs = []
        for cut in batchCuts:
            rIDs.append(cut.supervisions[0].recording_id)
            print(cut.supervisions[0].to_dict())

        if i < 10:
            print(i, len(audio_lengths), audio_lengths)
            print(rIDs)
            print("audios size:", audios.size(), ", audio_lens size:", audio_lengths.size())
            i += 1
            model_out = model.forward(
                input_signal=audios, 
                input_signal_length=audio_lengths,
                # segment_lens=batch['segment_lens'] if isfalse(args.do_not_pass_segment_lens) else None,
                # return_cross_utterance_attention=True if args.return_attention else None 
            ) 
            log_probs, _, encoded_len = model_out[:3]
            batch_labels = torch.argmax(log_probs, dim = 2)
            for cutNo, cut in enumerate(batchCuts):
                start_time = 0
                lastLabel = batch_labels[0].item()
                supID = 0
                for frame, label in enumerate(labels):
                    if label != lastLabel:
                        supervisions.append(
                            SupervisionSegment(
                                id = rIDs[cutNo] + str(supID), 
                                recording_id = rIDs[cutNo],
                                start = start_time * timeCorrection,
                                duration = (frame - start_time) * timeCorrection
                            )
                )




            # if save_labels:
                # labels.extend(batch_label[0:labels_len] for batch_label, labels_len in zip(batch_labels, encoded_len))
                # batch_shape = batch_labels.size()
                # print("log_probs size:", log_probs.size(), ", batch_shape:", batch_shape, ", labels size:", labels.size())
                # if batch_shape[1] > labels.size()[1]:
                #     labels = torch.nn.functional.pad(labels, (0, batch_shape[1] - labels.size()[1], 0, 0))
                # else:
                #     batch_labels = torch.nn.functional.pad(batch_labels, (0, labels.size()[1] - batch_shape[1], 0, 0))
                # labels[done_so_far:done_so_far + batch_shape[0]] = torch.argmax(log_probs)
                # labels_lens[done_so_far:done_so_far + len(encoded_len)] = encoded_len
                # done_so_far += len(encoded_len)
                # if i < 10:
                #     print(labels[done_so_far - 1].size())
                #     print(labels_lens[done_so_far - 1])
                #     print(len(labels), len(labels_lens), done_so_far)
                #     # print(labels)
                #     # print(labels_lens)
                #     i += 1



        else:
            quit()


    #     print(i)
    #     i += 1
    #     audio, audio_len = collate_audio(CutSet([cut]))
        

    # print(max_len_samps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser() 

    parser.add_argument('--load_pretrained', action='store_true')
    parser.add_argument('--pretrained', type=str, default='stt_en_conformer_ctc_small') # stt_en_conformer_ctc_large stt_en_conformer_transducer_large
    parser.add_argument('--model_config', type=str, default='/home/acp20rm/exp/nemo_ex/asr_training_scripts/model_configs/conformer_sc_ctc_bpe_small.yaml') 

    parser.add_argument('--tokenizer', type=str, default='/home/acp20rm/exp/nemo_ex/asr_training_scripts/ami/tokenizer_spe_bpe_v128', help='path to tokenizer dir')
    parser.add_argument('--max_duration', type=float, default=60, help='max duration of audio in seconds')
    # parser.add_argument('--num_meetings', type=int, default=1, help='number of meetings per batch')

    parser.add_argument('--log_file', type=str, default='eval_log.txt')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/sc-ctc/')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_209_id_5.pt')
    parser.add_argument('--beam_size', type=int, default=100)
    parser.add_argument('-lm', '--language_model', type=str, default='', help='arpa n-gram model for decoding')#./ngrams/3gram-6mix.arpa
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--sweep', action='store_true', help='run wandb search for language model weight')
    # parser.add_argument('-sc','--self_conditioned', action='store_false', help='use self-conditioned model (STORE FALSE)')

    # parser.add_argument('--config_from_checkpoint_dir', action='store_false', help='load config from checkpoint dir') ##chane
    # parser.add_argument('-cer', '--cer', action='store_true', help='compute CER instead of WER')

    # parser.add_argument('--return_attention', action='store_true', help='return attention')
    # parser.add_argument('--save_attention', action='store_true', help='save attention')
    # parser.add_argument('--save_attn_dir', type=str, default='./attns')

    # parser.add_argument('-gap','--gap', default=0.1, type=float, help='gap between utterances when concatenating')

    # parser.add_argument('--single_speaker_with_gaps', action='store_true', help='if set, utterances will contain 1 speaker and additional gaps of speaker_gap will be added if there is a speaker change between two utternces of the same speaker')
    # parser.add_argument('--speaker_gap', type=float, default=1.0, help='for use with single_speaker_with_gaps, will add this many seconds of silence between utterances of the same speaker when there is a speaker change in between them')

    # parser.add_argument('--concat_samples', action='store_true', help='if set, will concat cuts from same meeting instead of stacking them')
    # parser.add_argument('--split_speakers', action='store_true', help='if set, wont concat samples from different speakers, (concat_samples must be enabled)')
    # parser.add_argument('-dnpsl','--do_not_pass_segment_lens', action='store_true', help='if set, will not pass segment lens to the model, used with concat_samples for single segment models')

    # parser.add_argument('-sclite','--sclite', action='store_true', help='if set, will eval with sclite')
    # parser.add_argument('-sclite_dir', '--sclite_dir', type=str, default='./trns')

    # parser.add_argument('-save','--save_outputs', default='', type=str, help='save outputs to file')
    parser.add_argument('-save_labels', default='/home/acp20rm/data/ami/labels/test/', type=str, help='save predicted frame labels for use as exemplars')

   
    args = parser.parse_args()

    if args.checkpoint != '':
        args.checkpoint = os.path.join(args.checkpoint_dir, args.checkpoint)

    # if args.config_from_checkpoint_dir == True:
    #     dir_contents = os.listdir(args.checkpoint_dir)
    #     config = [el for el in dir_contents if el.endswith('.yaml')]
    #     assert len(config) == 1, 'Exactly one config file must be in checkpoint dir'
    #     args.model_config = os.path.join(args.checkpoint_dir, config[0])
    #     print(f'Loading config from checkpoint dir: {args.model_config}')


    # if args.load_logits_location != '' and os.path.exists(args.load_logits_location) == False:
    #     raise ValueError(f'{args.load_logits_location} does not exist')

    # if args.save_text == True and os.path.exists('./txt_outputs') == False:
    #     os.mkdir('./txt_outputs')
  
    main(args)