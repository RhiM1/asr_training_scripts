import math
from multiprocessing import context
from pyexpat import features
import json

from tracemalloc import start
from types import new_class

from numpy import dtype
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from getMFCCs import ID_to_utt
import torch
from torch import nn, threshold
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import time
from lhotse import CutSet, Seconds
from lhotse.dataset.sampling.base import CutSampler, TimeConstraint
from lhotse.dataset.sampling.data_source import DataSource
from lhotse.dataset.sampling.simple import SimpleCutSampler
# from tools import eval_dataloader
from tqdm import tqdm
from lhotse.dataset.collation import collate_audio
from tools import TokenizerCollator
from non_iid_dataloader import get_data_loader, get_eval_dataloader

class MinimalDataset(torch.utils.data.Dataset):
  def __init__(self, tokenizer: TokenizerCollator):
    self.tokenizer = tokenizer

  def __getitem__(self, cuts: CutSet) -> dict:
    # cuts = cuts.sort_by_duration()
    audios, audio_lens = collate_audio(cuts)
    tokens, token_lens = self.tokenizer(cuts)
    return {
        "audio": audios,
        "audio_lens": audio_lens,
        "tokens": tokens,
        "token_lens": token_lens,
    }

# def load_dataloader(cuts, tokenizer, max_duration:int, shuffle:bool=True):
def load_dataloader(cuts, tokenizer, batch_size:int, shuffle:bool=True):
    '''
    Example usage:
    - Obtain corpus
    \n
    ami_dict = load_corpus()
    train_dl = load_dataloader(ami_dict['train'], tokenizer, max_duration=360, shuffle=True)
    \n
    - tokenizer should be a sentencepiece tokenizer
    '''
    collator = TokenizerCollator(tokenizer)
    dataset = MinimalDataset(collator)
    sampler = SimpleCutSampler(cuts, shuffle=False, max_cuts=batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
    
    return dataloader

class EvaluationDataset(torch.utils.data.Dataset):
    '''
    Dataset for use when evaluating the model
    Therefore we need to return audio, 
    and text but no tokens!
    '''
    @staticmethod
    def __getitem__(cuts: CutSet) -> dict:
        # cuts = cuts.sort_by_duration()
        audios, audio_lens = collate_audio(cuts)
        return {
            "audio": audios,
            "audio_lens": audio_lens,
            "text": [" ".join(supervision.text for supervision in cut.supervisions) for cut in cuts],
        }

def eval_dataloader(cuts, batch_size:int, shuffle:bool=False):
    dataset = EvaluationDataset()
    sampler = SimpleCutSampler(cuts, shuffle=False, max_cuts=batch_size)
    dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler)
    return dataloader

class ami_ex_dataset(Dataset):
    def __init__(
        self,
        args,
        cuts: CutSet, 
        # labels,
        # labels_lengths,
        model = None, 
        tokenizer = None,
        batch_size = 50,
        load_labels_file = '',
        load_labels_lens_file = '',
        load_features_file = '',
        load_features_lens_file = '',
        save_labels_file = '',
        save_labels_lens_file = '',
        save_features_file = '',
        save_features_lens_file = ''
        ):
        
        with torch.no_grad():
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device)

            # self.cuts = cuts
            # self.labels = labels
            # self.labels_lens = labels_lengths
            # self.model = model
            # self.features_dir = features
            # self.save_featuers_dir = save_features_dir

            if load_labels_file != '':
                get_labels = True
            else:
                get_labels = False

            if load_features_file != '':
                get_features = True
            else:
                get_features = False

            dl = get_data_loader(
                split=cuts, 
                tokenizer=tokenizer, 
                shuffle=False, 
                max_duration=args.micro_batch_duration, 
                num_workers=args.num_workers, 
                batch_size=args.micro_batch_number, 
                concat_samples=args.concat_samples,
                split_speakers=args.split_speakers,
                gap=args.gap,
                speaker_gap=args.speaker_gap,
                single_speaker_with_gaps=args.single_speaker_with_gaps,
            )


            pbar = tqdm(dl, total=len(dl.sampler.data_source))

            for batchID, batch in enumerate(pbar):
                current_batch_size = len(batch['audio_lens'][0])
                pbar.update(current_batch_size)

                if get_labels:
                    self.get_labels_from_cuts(batch, model)


                if load_features_file == '':
                    self.get_features_from_cutset(dl, model, batch_size, True, tokenizer)
                else:
                    self.get_features_from_cutset(cuts, model, batch_size, False, tokenizer)
                    self.load_features_from_file(load_features_file, load_features_lens_file)

                if save_features_file != '' and load_features_file == '':
                    # There's a save file and the features have been computer 
                    # (rather than loaded)
                    self.save_features_to_file(save_features_file, save_features_lens_file)

    def get_features_from_batch(self, batch):
        pass

    def get_tokens_from_batch(self, batch):
        pass

    def get_text_from_batch(self, batch):
        pass
        
        
    def get_features_from_cutset(self, cuts, model, batch_size, get_feats, tokenizer: TokenizerCollator = None):
        '''
        Needs to return:
        - features (list of tensors, each of size uttLength)
        - features length (tensor of size DataSetLenth)
        '''



            # if tokenizer is None:
            #     self.has_tokens = False
            # else:
            #     self.has_tokens = True

            if tokenizer is None:
                get_tokens = False
                feats_dl = eval_dataloader(cuts, batch_size=batch_size, shuffle = False)
                tokens = None
                tokens_lens = None
                texts = []
                self.has_tokens = False
            else:
                get_tokens = True
                feats_dl = load_dataloader(cuts, tokenizer, batch_size, False)
                texts = None
                self.has_tokens = True



            # features = torch.empty(0)
            start_idx = 0
            # i = 0
                # print(batchID)
                # print("Batch", batchID)
                # i += 1

                if get_feats:
                    audios = batch['audio'].reshape(-1, batch['audio'].shape[-1]).to(device)
                    # pbar.update(audios.shape[0])
                    
                    audio_lengths = batch['audio_lens'].reshape(-1).to(device)
                    # print("audios size:\n", audios.size())
                    # print("audio_lengths size:\n", audio_lengths.size())

                    feats, feats_lengths = model.preprocessor(
                        input_signal=audios, 
                        length=audio_lengths
                    )
                    current_batch_size = len(feats_lengths)
                    batch_max_feats_len = torch.max(feats_lengths)

                    # print("batch max len:", batch_max_len)
                    if batchID == 0:
                        # print("start_idx:", start_idx, ", batch_size:", current_batch_size)
                        # First batch, initialise
                        features_lengths = torch.empty(len(cuts), dtype = torch.long)
                        features = torch.empty((len(cuts), 80, batch_max_feats_len))
                        features[0:current_batch_size] = feats.to('cpu')
                        features_lengths[0:current_batch_size] = feats_lengths.to('cpu')
                        max_len = batch_max_feats_len

                    elif batch_max_feats_len > max_len:
                        # Bigger features than any previous batch - 
                        # pad the full features set to match
                        max_len = batch_max_feats_len
                        features = torch.nn.functional.pad(features, (0, max_len - features.size()[2], 0, 0, 0, 0))
                        features[start_idx:start_idx + current_batch_size] = feats.to('cpu')
                        features_lengths[start_idx:start_idx + current_batch_size] = feats_lengths.to('cpu')

                    else:
                        # Smaller or same size features as largest previous batches - 
                        # pad this batch's features to match
                        feats = torch.nn.functional.pad(feats, (0, max_len - feats.size()[2], 0, 0, 0, 0))
                        features[start_idx:start_idx + current_batch_size] = feats.to('cpu')
                        features_lengths[start_idx:start_idx + current_batch_size] = feats_lengths.to('cpu')


                if get_tokens:
                    # Train / val dataset, need to get tokens
                    tokes = batch['tokens'][0]
                    tokes_lens = batch['token_lens']
                    batch_max_tokens_len = tokes.size()[1]
                    if batchID == 0:
                        tokens = torch.empty((len(cuts), batch_max_tokens_len))
                        tokens[0:current_batch_size] = tokes

                        tokens_lens = torch.empty(len(cuts))
                        tokens_lens[0:current_batch_size] = tokes_lens
                        max_tokens_len = batch_max_tokens_len
                    elif batch_max_tokens_len > max_tokens_len:
                        # Bigger tokens than any previous batch - 
                        # pad the full tokens set to match
                        max_tokens_len = batch_max_tokens_len
                        tokens = torch.nn.functional.pad(tokens, (0, max_tokens_len - tokens.size()[1], 0, 0))
                        tokens[start_idx:start_idx + current_batch_size] = tokes
                        tokens_lens[start_idx:start_idx + current_batch_size] = tokes_lens
                    else:
                        # Smaller or same size tokens as largest previous batches - 
                        # pad this batch's tokens to match
                        tokes = torch.nn.functional.pad(tokes, (0, max_tokens_len - tokes.size()[1], 0, 0))
                        tokens[start_idx:start_idx + current_batch_size] = tokes
                        tokens_lens[start_idx:start_idx + current_batch_size] = tokes_lens
                else:
                    # Evaluation datset, just get the text for each utterance
                    for text in batch['text']:
                        texts.extend(text)

                start_idx += current_batch_size

            if get_feats:
                self.features = features
                self.features_lens = features_lengths
            self.tokens = tokens
            self.tokens_lens = tokens_lens
            self.texts = texts


    def load_features_from_file(self, load_features_file, load_feats_lens_file):
        '''
        Needs to return:
        - features (list of tensors, each of size uttLength)
        '''
        self.features = torch.load(load_features_file)
        self.features_lens = torch.load(load_feats_lens_file)

    def save_features_to_file(self, save_features_file, save_features_lens_file):
        '''
        Save the list of feaures for future use
        '''
        torch.save(self.features, save_features_file)
        torch.save(self.features_lens, save_features_lens_file)


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Needs to return:
        - features (list of tensors, each size uttLength x numFeatures)
        - labels (list of tensors, each of size uttLength)
        - features length (tensor of size DataSetLenth)
        - labels length (tensor of size DataSetLenth)
            may differ from features length due to covolutional sub-sampling
        '''
        if self.has_tokens:
            return {
                'features': self.features[idx], 
                'features_lens': self.features_lens[idx], 
                'labels': self.labels[idx], 
                'labels_lens': self.labels_lens[idx],
                'tokens': self.tokens[idx],
                'tokens_lens': self.tokens_lens[idx],
            }
        else:
            return {
                'features': self.features[idx], 
                'features_lens': self.features_lens[idx], 
                'labels': self.labels[idx], 
                'labels_lens': self.labels_lens[idx],
                'texts': self.texts[idx]
            }
    


