# !usr/bin/env python
# -*- coding:utf-8 _*-
"""
@Author: Huiqiang Xie
@File: performance.py
@Time: 2021/4/1 11:48
"""
import os
import json
import torch
import argparse
import numpy as np
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from utils import BleuScore, SNR_to_noise, greedy_decode, SeqtoText, Channels, create_masks, PowerNormalize
from tqdm import tqdm
from sklearn.preprocessing import normalize
# from bert4keras.backend import keras
# from bert4keras.models import build_bert_model
# from bert4keras.tokenizers import Tokenizer
from w3lib.html import remove_tags
# Import SemRect modules
from SemRect import SemRect
from deepsc_semrect import DeepSCWithSemRect

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', default='europarl/train_data.pkl', type=str)
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--checkpoint-path', default='checkpoints/deepsc-Rayleigh', type=str)
parser.add_argument('--channel', default='Rayleigh', type=str)
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type = int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=64, type=int)
parser.add_argument('--epochs', default=2, type = int)
parser.add_argument('--bert-config-path', default='bert/cased_L-12_H-768_A-12/bert_config.json', type = str)
parser.add_argument('--bert-checkpoint-path', default='bert/cased_L-12_H-768_A-12/bert_model.ckpt', type = str)
parser.add_argument('--bert-dict-path', default='bert/cased_L-12_H-768_A-12/vocab.txt', type = str)
# Add adversarial attack parameters
parser.add_argument('--test-attacks', action='store_true', help='Enable adversarial attack testing')
parser.add_argument('--epsilon', default=0.1, type=float, help='FGSM attack strength')
parser.add_argument('--semrect-checkpoint', default=None, type=str, help='Path to DeepSC+SemRect checkpoint')
parser.add_argument('--semrect-latent-dim', default=100, type=int, help='Latent dimension for SemRect')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# using pre-trained model to compute the sentence similarity
# class Similarity():
#     def __init__(self, config_path, checkpoint_path, dict_path):
#         self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
#         self.model = keras.Model(inputs=self.model1.input,
#                                  outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
#         # build tokenizer
#         self.tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
#     def compute_similarity(self, real, predicted):
#         token_ids1, segment_ids1 = [], []
#         token_ids2, segment_ids2 = [], []
#         score = []
#
#         for (sent1, sent2) in zip(real, predicted):
#             sent1 = remove_tags(sent1)
#             sent2 = remove_tags(sent2)
#
#             ids1, sids1 = self.tokenizer.encode(sent1)
#             ids2, sids2 = self.tokenizer.encode(sent2)
#
#             token_ids1.append(ids1)
#             token_ids2.append(ids2)
#             segment_ids1.append(sids1)
#             segment_ids2.append(sids2)
#
#         token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
#         token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')
#
#         segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
#         segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')
#
#         vector1 = self.model.predict([token_ids1, segment_ids1])
#         vector2 = self.model.predict([token_ids2, segment_ids2])
#
#         vector1 = np.sum(vector1, axis=1)
#         vector2 = np.sum(vector2, axis=1)
#
#         vector1 = normalize(vector1, axis=0, norm='max')
#         vector2 = normalize(vector2, axis=0, norm='max')
#
#         dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
#         a = np.diag(np.matmul(vector1, vector1.T))  # a*a
#         b = np.diag(np.matmul(vector2, vector2.T))
#
#         a = np.sqrt(a)
#         b = np.sqrt(b)
#
#         output = dot / (a * b)
#         score = output.tolist()
#
#         return score


def performance(args, SNR, net):
    # similarity = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
    bleu_score_1gram = BleuScore(1, 0, 0, 0)

    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                               pin_memory=True, collate_fn=collate_data)

    StoT = SeqtoText(token_to_idx, end_idx)
    score = []
    score2 = []
    net.eval()
    with torch.no_grad():
        for epoch in range(args.epochs):
            Tx_word = []
            Rx_word = []

            for snr in tqdm(SNR):
                word = []
                target_word = []
                noise_std = SNR_to_noise(snr)

                for sents in test_iterator:

                    sents = sents.to(device)
                    # src = batch.src.transpose(0, 1)[:1]
                    target = sents

                    out = greedy_decode(net, sents, noise_std, args.MAX_LENGTH, pad_idx,
                                        start_idx, args.channel)

                    sentences = out.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, sentences))
                    word = word + result_string

                    target_sent = target.cpu().numpy().tolist()
                    result_string = list(map(StoT.sequence_to_text, target_sent))
                    target_word = target_word + result_string

                Tx_word.append(word)
                Rx_word.append(target_word)

            bleu_score = []
            sim_score = []
            for sent1, sent2 in zip(Tx_word, Rx_word):
                # 1-gram
                bleu_score.append(bleu_score_1gram.compute_blue_score(sent1, sent2)) # 7*num_sent
                # sim_score.append(similarity.compute_similarity(sent1, sent2)) # 7*num_sent
            bleu_score = np.array(bleu_score)
            bleu_score = np.mean(bleu_score, axis=1)
            score.append(bleu_score)

            # sim_score = np.array(sim_score)
            # sim_score = np.mean(sim_score, axis=1)
            # score2.append(sim_score)

    score1 = np.mean(np.array(score), axis=0)
    # score2 = np.mean(np.array(score2), axis=0)

    return score1#, score2


def greedy_decode_with_semrect(model, src, noise_std, max_len, pad_idx, start_symbol, channel_type, 
                              use_semrect=False, epsilon=0.1, attack=False):
    """
    Greedy decoding with SemRect and adversarial attack support
    """
    channels = Channels()
    model.eval()
    
    src_mask = (src == pad_idx).unsqueeze(-2).type(torch.FloatTensor).to(device)
    
    # Step 1: Get semantic representation (with or without attack)
    with torch.no_grad():
        # Regular encoder processing
        enc_output = model.deepsc.encoder(src, src_mask)
        channel_enc = model.deepsc.channel_encoder(enc_output)
        tx_sig = PowerNormalize(channel_enc)
        
        # Channel effects
        if channel_type == 'AWGN':
            rx_sig = channels.AWGN(tx_sig, noise_std)
        elif channel_type == 'Rayleigh':
            rx_sig = channels.Rayleigh(tx_sig, noise_std)
        elif channel_type == 'Rician':
            rx_sig = channels.Rician(tx_sig, noise_std)
            
        # Channel decoder to get semantic representation
        semantic_repr = model.deepsc.channel_decoder(rx_sig)
    
    # Step 2: Apply attack if requested
    if attack:
        # Enable gradients for attack
        semantic_repr = semantic_repr.detach().clone().requires_grad_(True)
        
        # Create a dummy target - goal is to maximize loss (misclassification)
        batch_size = src.size(0)
        dummy_trg = torch.ones(batch_size, 1, dtype=torch.long).to(device) * start_symbol
        
        # Forward pass with the semantic representation
        dummy_out = model.deepsc.decoder(dummy_trg, semantic_repr, None, src_mask)
        dummy_logits = model.deepsc.dense(dummy_out)
        
        # This will be used to create an FGSM-style attack
        # Pick a random target different from the predicted class
        pred_class = dummy_logits.argmax(dim=-1)
        random_target = (pred_class + torch.randint(1, pred_class.shape[-1], pred_class.shape).to(device)) % pred_class.shape[-1]
        
        # Compute loss
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(dummy_logits.squeeze(1), random_target.squeeze(1))
        
        # Compute gradients
        loss.backward()
        
        # Create adversarial example
        with torch.no_grad():
            adv_semantic = semantic_repr + epsilon * semantic_repr.grad.sign()
            semantic_repr = adv_semantic
    
    # Step 3: Apply SemRect calibration if enabled
    if use_semrect:
        with torch.no_grad():
            semantic_repr = model.semrect.calibrate(semantic_repr)
    
    # Step 4: Decode the sentence
    with torch.no_grad():
        memory = semantic_repr
        ys = torch.ones(src.size(0), 1).fill_(start_symbol).type(torch.long).to(device)
        
        for i in range(max_len - 1):
            # Generate next token
            out = model.deepsc.decoder(ys, memory, None, src_mask)
            prob = model.deepsc.dense(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.unsqueeze(1)
            
            # Concatenate with output sequence
            ys = torch.cat([ys, next_word], dim=1)
            
            # Stop if we hit EOS token
            if (next_word == pad_idx).all():
                break
                
    return ys


def test_adversarial_attacks(args, SNR, std_model, semrect_model):
    """
    Test both models against adversarial attacks
    """
    print("Testing models against adversarial attacks...")
    bleu_score_1gram = BleuScore(1, 0, 0, 0)
    
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                             pin_memory=True, collate_fn=collate_data)
    
    StoT = SeqtoText(token_to_idx, end_idx)
    
    # Results dictionary
    results = {
        'snr': SNR,
        'normal': {
            'no_attack': [],
            'under_attack': []
        },
        'semrect': {
            'no_attack': [],
            'under_attack': []
        }
    }
    
    for snr in tqdm(SNR, desc="Testing SNRs"):
        noise_std = SNR_to_noise(snr)
        
        # Store decoded sentences for BLEU calculation
        original_sentences = []
        std_sentences_clean = []
        std_sentences_attack = []
        semrect_sentences_clean = []
        semrect_sentences_attack = []
        
        # Process each batch
        for sents in tqdm(test_iterator, desc=f"Processing batches (SNR={snr})"):
            sents = sents.to(device)
            target = sents  # Original sentences
            
            # Get original sentences for comparison
            target_sent = target.cpu().numpy().tolist()
            original_text = list(map(StoT.sequence_to_text, target_sent))
            original_sentences.extend(original_text)
            
            # Standard model - Clean decoding
            out_std_clean = greedy_decode(std_model, sents, noise_std, args.MAX_LENGTH, 
                                         pad_idx, start_idx, args.channel)
            sentences_std_clean = out_std_clean.cpu().numpy().tolist()
            text_std_clean = list(map(StoT.sequence_to_text, sentences_std_clean))
            std_sentences_clean.extend(text_std_clean)
            
            # SemRect model - Clean decoding
            out_semrect_clean = greedy_decode_with_semrect(
                semrect_model, sents, noise_std, args.MAX_LENGTH, 
                pad_idx, start_idx, args.channel, use_semrect=True, attack=False
            )
            sentences_semrect_clean = out_semrect_clean.cpu().numpy().tolist()
            text_semrect_clean = list(map(StoT.sequence_to_text, sentences_semrect_clean))
            semrect_sentences_clean.extend(text_semrect_clean)
            
            # Standard model - Under attack
            out_std_attack = greedy_decode_with_semrect(
                semrect_model, sents, noise_std, args.MAX_LENGTH, 
                pad_idx, start_idx, args.channel, use_semrect=False, attack=True, epsilon=args.epsilon
            )
            sentences_std_attack = out_std_attack.cpu().numpy().tolist()
            text_std_attack = list(map(StoT.sequence_to_text, sentences_std_attack))
            std_sentences_attack.extend(text_std_attack)
            
            # SemRect model - Under attack
            out_semrect_attack = greedy_decode_with_semrect(
                semrect_model, sents, noise_std, args.MAX_LENGTH, 
                pad_idx, start_idx, args.channel, use_semrect=True, attack=True, epsilon=args.epsilon
            )
            sentences_semrect_attack = out_semrect_attack.cpu().numpy().tolist()
            text_semrect_attack = list(map(StoT.sequence_to_text, sentences_semrect_attack))
            semrect_sentences_attack.extend(text_semrect_attack)
        
        # Calculate BLEU scores
        bleu_std_clean = np.mean(bleu_score_1gram.compute_blue_score(original_sentences, std_sentences_clean))
        bleu_std_attack = np.mean(bleu_score_1gram.compute_blue_score(original_sentences, std_sentences_attack))
        bleu_semrect_clean = np.mean(bleu_score_1gram.compute_blue_score(original_sentences, semrect_sentences_clean))
        bleu_semrect_attack = np.mean(bleu_score_1gram.compute_blue_score(original_sentences, semrect_sentences_attack))
        
        # Store results
        results['normal']['no_attack'].append(bleu_std_clean)
        results['normal']['under_attack'].append(bleu_std_attack)
        results['semrect']['no_attack'].append(bleu_semrect_clean)
        results['semrect']['under_attack'].append(bleu_semrect_attack)
        
        # Print current SNR results
        print(f"\nResults for SNR = {snr} dB:")
        print(f"Standard model (clean): {bleu_std_clean:.4f}")
        print(f"Standard model (attack): {bleu_std_attack:.4f}")
        print(f"SemRect model (clean): {bleu_semrect_clean:.4f}")
        print(f"SemRect model (attack): {bleu_semrect_attack:.4f}")
        print(f"Improvement under attack: {((bleu_semrect_attack - bleu_std_attack) / bleu_std_attack * 100):.2f}%")
    
    return results


def plot_attack_results(results):
    """
    Plot the results of adversarial attack testing
    """
    import matplotlib.pyplot as plt
    
    snr = results['snr']
    std_clean = results['normal']['no_attack']
    std_attack = results['normal']['under_attack']
    semrect_clean = results['semrect']['no_attack']
    semrect_attack = results['semrect']['under_attack']
    
    plt.figure(figsize=(10, 6))
    plt.plot(snr, std_clean, 'b-o', label='Standard DeepSC (No Attack)')
    plt.plot(snr, std_attack, 'b--x', label='Standard DeepSC (Under Attack)')
    plt.plot(snr, semrect_clean, 'r-o', label='DeepSC+SemRect (No Attack)')
    plt.plot(snr, semrect_attack, 'r--x', label='DeepSC+SemRect (Under Attack)')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('BLEU Score')
    plt.title(f'DeepSC vs DeepSC+SemRect Under Adversarial Attack (ε={args.epsilon})')
    plt.grid(True)
    plt.legend()
    
    # Save and show
    plt.savefig('attack_results.png')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()
    SNR = [0,3,6,9,12,15,18]

    args.vocab_file = '/import/antennas/Datasets/hx301/' + args.vocab_file
    vocab = json.load(open(args.vocab_file, 'rb'))
    token_to_idx = vocab['token_to_idx']
    idx_to_token = dict(zip(token_to_idx.values(), token_to_idx.keys()))
    num_vocab = len(token_to_idx)
    pad_idx = token_to_idx["<PAD>"]
    start_idx = token_to_idx["<START>"]
    end_idx = token_to_idx["<END>"]

    # Load standard DeepSC model
    deepsc = DeepSC(args.num_layers, num_vocab, num_vocab,
                    num_vocab, num_vocab, args.d_model, args.num_heads,
                    args.dff, 0.1).to(device)

    model_paths = []
    for fn in os.listdir(args.checkpoint_path):
        if not fn.endswith('.pth'): continue
        idx = int(os.path.splitext(fn)[0].split('_')[-1])  # read the idx of image
        model_paths.append((os.path.join(args.checkpoint_path, fn), idx))

    model_paths.sort(key=lambda x: x[1])  # sort the image by the idx

    model_path, _ = model_paths[-1]
    checkpoint = torch.load(model_path)
    deepsc.load_state_dict(checkpoint)
    print('Standard DeepSC model loaded!')

    # Check if testing against adversarial attacks
    if args.test_attacks:
        if args.semrect_checkpoint is None:
            print("Error: --semrect-checkpoint must be provided when using --test-attacks")
            exit(1)
            
        # Load DeepSC+SemRect model
        print(f"Loading DeepSC+SemRect model from {args.semrect_checkpoint}")
        deepsc_semrect = DeepSCWithSemRect(
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
        
        # Load checkpoint
        checkpoint_semrect = torch.load(args.semrect_checkpoint)
        deepsc_semrect.load_state_dict(checkpoint_semrect)
        print('DeepSC+SemRect model loaded!')
        
        # Run adversarial attack testing
        attack_results = test_adversarial_attacks(args, SNR, deepsc, deepsc_semrect)
        
        # Plot results
        plot_attack_results(attack_results)
        
        # Save results to file
        np.save('attack_results.npy', attack_results)
        print("Attack test results saved to attack_results.npy")
    else:
        # Standard performance evaluation
        bleu_score = performance(args, SNR, deepsc)
        print(bleu_score)

    #similarity.compute_similarity(sent1, real)
