'''
General utility functions for model training
'''
import torch
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.models.rnnt_bpe_models import EncDecRNNTBPEModel
from nemo.collections.asr.models.scctc_bpe_models import EncDecSCCTCModelBPE
from nemo.collections.asr.models.scctc_models import EncDecSCCTCModel
from nemo.collections.asr.models.ex_scctc_models import EncDecExSCCTCModel
import os
import numpy as np
from omegaconf.omegaconf import OmegaConf
import json

def load_checkpoint(args, model, optim=None):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(checkpoint['model_state_dict'].keys())
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Error loading model state_dict: {e}, loading attempted with strict=False')
    if 'no_load_optim' in args.__dict__ and args.no_load_optim == True:
        print('Not loading optimizer')
    elif optim is not None:
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint else None
    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {epoch}, Validation loss: {val_loss}')
    return epoch, val_loss

def get_ex_model(args):
    
    ex_cfg = OmegaConf.load(args.ex_model_config)
    exModel = EncDecSCCTCModelBPE(ex_cfg['model'])
    print(f'Loaded ex-model from config file {args.ex_model_config}')
    checkpoint_path = args.ex_checkpoint_dir + args.ex_checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    # print(checkpoint['model_state_dict'].keys())
    try:
        exModel.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        exModel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Error loading model state_dict: {e}, loading attempted with strict=False')
    # epoch = checkpoint['epoch'] if 'epoch' in checkpoint else 0
    # val_loss = checkpoint['val_loss'] if 'val_loss' in checkpoint else None
    print(f'Loaded exemplar checkpoint from {checkpoint_path}')
    # print(f'Epoch: {epoch}, Validation loss: {val_loss}')
    # return epoch, val_loss
    return exModel

def load_nemo_checkpoint(args, model, optim):
    checkpoint_path = args.checkpoint
    print(checkpoint_path)
    checkpoint = torch.load(checkpoint_path)
    print(checkpoint['state_dict'].keys())
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer_states'][0])
    epoch = checkpoint['epoch']
    val_loss = None
    print(f'Loaded checkpoint from {checkpoint_path}')
    print(f'Epoch: {epoch}, Validation loss: {val_loss}')
    return epoch, val_loss


def load_model(args):
    print(args.load_pretrained)
    if args.load_pretrained == True:
        model = EncDecCTCModelBPE.from_pretrained(args.pretrained)
        if args.tokenizer != '':
            model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = EncDecCTCModelBPE(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model

def load_sc_model(args):
    print(args.load_pretrained)
    if args.load_pretrained == True:
        model = EncDecSCCTCModel.from_pretrained(args.pretrained)
        if args.tokenizer != '':
            model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = EncDecSCCTCModel(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model

def load_ex_sc_model(args):
    print(args.load_pretrained)
    if args.load_pretrained == True:
        model = EncDecExSCCTCModel.from_pretrained(args.pretrained)
        if args.tokenizer != '':
            model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = EncDecExSCCTCModel(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model

def load_transducer_model(args):
    print(args.load_pretrained)
    if args.load_pretrained == True:
        model = EncDecRNNTBPEModel.from_pretrained(args.pretrained)
        # if args.tokenizer != '':
        #     model.change_vocabulary(new_tokenizer_dir=args.tokenizer, new_tokenizer_type='bpe')
        return model
    else:
        cfg = OmegaConf.load(args.model_config)
        model = EncDecRNNTBPEModel(cfg['model'])
        print(f'Loaded model from config file {args.model_config}')
        return model

def squeeze_batch_and_to_device(batch, device):
    input_signal = batch['audio'].reshape(-1, batch['audio'].shape[-1]).to(device)
    input_signal_lengths = batch['audio_lens'].reshape(-1).to(device)
    targets = batch['tokens'].reshape(-1, batch['tokens'].shape[-1]).to(device)
    target_lengths = batch['token_lens'].reshape(-1).to(device)
    batch_size = input_signal.shape[0]
    return input_signal, input_signal_lengths, targets, target_lengths, batch_size

def write_to_log(log_file, data):
    with open(log_file, 'a') as f:
        f.write(data)
        f.write('\n')


def save_checkpoint(args, model, optim, epoch, val_loss):
    path = os.path.join(args.checkpoint_dir, f'checkpoint_{epoch}_id_{np.random.randint(0,100)}.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'val_loss': val_loss
    }, path)
    print(f'Saved checkpoint to {path}')
    return path


def draw_text(text):
    print(f'\n \n ------------------------------------------------- ')
    print(f' ----------------- {text} ----------------- ')
    print(f' ------------------------------------------------- \n \n')


def load_schedular_data(args):
    with open(args.schedular_data, 'r') as f:
        data = json.load(f)
    return data['max_lr'], data['min_lr'], data['step_size']

def save_schedular_data(args):
    tosave = {
        'max_lr': args.max_lr,
        'min_lr': args.min_lr,
        'step_size': args.step_size
    }
    with open(args.schedular_data, 'w') as f:
        json.dump(tosave, f)

def decode_lm(logits_list, decoder, beam_width=100, encoded_lengths=None):
    decoded = []
    if encoded_lengths is None:
        encoded_lengths = [len(logits) for logits in logits_list]

    for logits, length in zip(logits_list, encoded_lengths):
        decoded.append(decoder.decode(logits[:length], beam_width=beam_width))

    return decoded