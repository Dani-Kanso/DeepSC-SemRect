# -*- coding: utf-8 -*-
"""
Created on Tue May 26 16:59:14 2020

@author: HQ Xie
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import numpy as np
from utils import SNR_to_noise, initNetParams, train_step, val_step, train_mi
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from models.mutual_info import Mine
from torch.utils.data import DataLoader
from tqdm import tqdm
# Import SemRect modules
from SemRect import SemRect
from deepsc_semrect import DeepSCWithSemRect  # Import the integrated model

parser = argparse.ArgumentParser()
#parser.add_argument('--data-dir', default='data/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str, help = 'Please choose AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
# Add SemRect parameters
parser.add_argument('--use-semrect', action='store_true', help='Enable SemRect integration')
parser.add_argument('--semrect-epochs', default=30, type=int, help='Number of epochs for SemRect training')
parser.add_argument('--semrect-latent-dim', default=100, type=int, help='Latent dimension for SemRect')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def validate(epoch, args, net):
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    net.eval()
    pbar = tqdm(test_iterator)
    total = 0
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            # Check if using regular DeepSC or DeepSCWithSemRect
            if isinstance(net, DeepSCWithSemRect):
                # Use SemRect-enabled validation
                noise_std = 0.1  # Fixed for validation
                output = net(sents, sents, noise_std, args.channel, use_semrect=True)
                # Calculate loss
                trg_inp = sents[:, :-1]
                trg_real = sents[:, 1:]
                loss = nn.CrossEntropyLoss(reduction='none')(
                    output.reshape(-1, output.size(-1)),
                    trg_real.reshape(-1)
                )
                # Apply mask
                mask = (trg_real != pad_idx).type_as(loss.data)
                loss = (loss * mask.reshape(-1)).sum() / mask.sum()
            else:
                # Standard DeepSC validation
                loss = val_step(net, sents, sents, 0.1, pad_idx,
                                criterion, args.channel)

            total += loss.item()
            pbar.set_description(
                'Epoch: {}; Type: VAL; Loss: {:.5f}'.format(
                    epoch + 1, loss.item()
                )
            )

    return total/len(test_iterator)


def train(epoch, args, net, mi_net=None):
    train_eur= EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    pbar = tqdm(train_iterator)

    noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))

    for sents in pbar:
        sents = sents.to(device)

        # Check if using regular DeepSC or DeepSCWithSemRect
        if isinstance(net, DeepSCWithSemRect):
            # Train with DeepSCWithSemRect
            optimizer.zero_grad()
            output = net(sents, sents, noise_std[0], args.channel, use_semrect=True)
            
            # Calculate loss
            trg_inp = sents[:, :-1]
            trg_real = sents[:, 1:]
            loss = nn.CrossEntropyLoss(reduction='none')(
                output.reshape(-1, output.size(-1)),
                trg_real.reshape(-1)
            )
            # Apply mask
            mask = (trg_real != pad_idx).type_as(loss.data)
            loss = (loss * mask.reshape(-1)).sum() / mask.sum()
            
            # Backward and optimize
            loss.backward()
            optimizer.step()
            
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss.item()
                )
            )
        elif mi_net is not None:
            mi = train_mi(net, mi_net, sents, 0.1, pad_idx, mi_opt, args.channel)
            loss = train_step(net, sents, sents, 0.1, pad_idx,
                              optimizer, criterion, args.channel, mi_net)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}; MI {:.5f}'.format(
                    epoch + 1, loss, mi
                )
            )
        else:
            loss = train_step(net, sents, sents, noise_std[0], pad_idx,
                              optimizer, criterion, args.channel)
            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss
                )
            )


def train_semrect(args, model):
    """Train the SemRect component of DeepSCWithSemRect"""
    print("Training SemRect component...")
    train_eur = EurDataset('train')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)
    
    # Train SemRect using the integrated method
    model.train_semrect(train_iterator, epochs=args.semrect_epochs, lr=1e-4)
    print("SemRect training completed!")


if __name__ == '__main__':
    # setup_seed(10)
    args = parser.parse_args()
    args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file
    """ preparing the dataset """
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]


    """ define model based on argument """
    if args.use_semrect:
        print("Using DeepSC with SemRect integration")
        model = DeepSCWithSemRect(
            num_layers=args.num_layers,
            src_vocab_size=num_vocab,
            trg_vocab_size=num_vocab,
            src_max_len=args.MAX_LENGTH,
            trg_max_len=args.MAX_LENGTH,
            d_model=args.d_model,
            num_heads=args.num_heads,
            dff=args.dff,
            dropout=0.1,
            semrect_latent_dim=args.semrect_latent_dim,
            device=device
        ).to(device)
        
        # First train the base DeepSC model
        initNetParams(model.deepsc)
    else:
        print("Using standard DeepSC model")
        model = DeepSC(args.num_layers, num_vocab, num_vocab,
                       num_vocab, num_vocab, args.d_model, args.num_heads,
                       args.dff, 0.1).to(device)
        initNetParams(model)

    """ define optimizer and loss function """
    mi_net = Mine().to(device)
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-4, betas=(0.9, 0.98), eps=1e-8, weight_decay = 5e-4)
    mi_opt = torch.optim.Adam(mi_net.parameters(), lr=1e-3)
    
    # Training loop for DeepSC
    print("Training DeepSC model...")
    for epoch in range(args.epochs):
        start = time.time()
        record_acc = 10

        train(epoch, args, model)
        avg_acc = validate(epoch, args, model)

        if avg_acc < record_acc:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            
            if args.use_semrect:
                # For DeepSCWithSemRect, save the entire model
                with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                    torch.save(model.state_dict(), f)
            else:
                # For standard DeepSC
                with open(args.checkpoint_path + '/checkpoint_{}.pth'.format(str(epoch + 1).zfill(2)), 'wb') as f:
                    torch.save(model.state_dict(), f)
                    
            record_acc = avg_acc
    
    # If using SemRect, train the SemRect component after DeepSC training
    if args.use_semrect:
        train_semrect(args, model)
        
        # Save the final model with trained SemRect
        final_path = args.checkpoint_path + '/final_with_semrect.pth'
        torch.save(model.state_dict(), final_path)
        print(f"Final model with trained SemRect saved to {final_path}")


    

        


