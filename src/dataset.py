'''
Pre-process the dataset
'''

import random
import torch
import numpy as np
import pandas as pd

from math import inf
from scipy import stats
from utils import random_label_assign
from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sentence_transformers import SentenceTransformer

'''Synthetic Noises: SN, ASN, IDN'''
def corrupt_dataset_SN(args, data):
    new_data = data.detach().clone()
    noise_ratio = args.noise_ratio * args.num_classes / (args.num_classes - 1)
    for i in range(len(new_data)):
        if random.random() > noise_ratio:
            continue
        else:
            new_data[i] = torch.randint(low=0, high=args.num_classes, size=(1, ))
    return new_data 

def corrupt_dataset_ASN(args, data):
    new_data = data.detach().clone()
    for i in range(len(new_data)):
        if random.random() > args.noise_ratio:
            continue
        else:
            new_data[i] = (new_data[i] + 1) % args.num_classes
    return new_data

def corrupt_dataset_IDN(args, inputs, labels):
    flip_distribution = stats.truncnorm((0-args.noise_ratio)/0.1, (1-args.noise_ratio)/0.1, loc=args.noise_ratio, scale=0.1)
    flip_rate = flip_distribution.rvs(len(labels))
    W = torch.randn(args.num_classes, inputs.shape[-1], args.num_classes).float()
    new_label = labels.detach().clone()
    for i in range(len(new_label)):
        p = inputs[i].float().view(1,-1).mm(W[labels[i].long()].squeeze(0)).squeeze(0)
        p[labels[i]] = -inf
        p = flip_rate[i] * torch.softmax(p, dim=0)
        p[labels[i]] += 1 - flip_rate[i]
        new_label[i] = torch.multinomial(p,1)
    return new_label 



def load_dataset(args):
    train_file_path = f"{args.dataset_path}/{args.dataset}_train.csv"
    valid_file_path = f"{args.dataset_path}/{args.dataset}_valid.csv"
    test_file_path = f"{args.dataset_path}/{args.dataset}_test.csv"

    train_file = pd.read_csv(train_file_path)
    valid_file = pd.read_csv(valid_file_path)
    test_file = pd.read_csv(test_file_path)

    train_true_labels = torch.tensor(train_file['true_label'].values, dtype=torch.long, device=args.device)
    valid_true_labels = torch.tensor(valid_file['true_label'].values, dtype=torch.long, device=args.device)
    test_true_labels = torch.tensor(test_file['true_label'].values, dtype=torch.long, device=args.device)

    train_noisy_labels = torch.tensor(train_file['noisy_label'].values, dtype=torch.long, device=args.device)
    valid_noisy_labels = torch.tensor(valid_file['noisy_label'].values, dtype=torch.long, device=args.device)
    test_noisy_labels = torch.tensor(test_file['noisy_label'].values, dtype=torch.long, device=args.device)

    train_input_sent = train_file['original_sent'].values
    valid_input_sent = valid_file['original_sent'].values
    test_input_sent = test_file['original_sent'].values

    if args.noise_type == "synthetic":
        if args.syn_type == "SN":
            del train_noisy_labels, valid_noisy_labels, test_noisy_labels
            train_noisy_labels = corrupt_dataset_SN(args, train_true_labels)
            valid_noisy_labels = corrupt_dataset_SN(args, valid_true_labels)
        elif args.syn_type == "ASN":
            del train_noisy_labels, valid_noisy_labels, test_noisy_labels
            train_noisy_labels = corrupt_dataset_ASN(args, train_true_labels)
            valid_noisy_labels = corrupt_dataset_ASN(args, valid_true_labels)

    elif args.noise_type in ["llm", "realworld"]:
        # if llm refuse to label a sample, 
        # we randomly assign a label to those missing answer sample
        train_noisy_labels = random_label_assign(args, train_noisy_labels)
        valid_noisy_labels = random_label_assign(args, valid_noisy_labels)

    return train_input_sent, train_true_labels, train_noisy_labels, valid_input_sent, valid_true_labels, valid_noisy_labels, test_input_sent, test_true_labels


def create_dataset(args):
    train_input_sent, train_true_labels, train_noisy_labels, valid_input_sent, valid_true_labels, \
            valid_noisy_labels, test_input_sent, test_true_labels = load_dataset(args)
    
    if args.dataset == "20news":
        MAX_LEN = 150
    elif args.dataset == "chemprot":
        MAX_LEN = 512
    else:
        MAX_LEN = 128

    # Encode train/test text
    # ===========================
    tokenizer = BertTokenizer.from_pretrained(args.plc, do_lower_case=True)
    train_input_ids = []
    train_attention_masks = []
    for sent in train_input_sent:
        encoded_sent = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            # This function also supports truncation and conversion
                            # to pytorch tensors, but we need to do padding, so we
                            # can't use these features :( .
                            max_length = MAX_LEN,          # Truncate all sentences.
                            #return_tensors = 'pt',     # Return pytorch tensors.
                    )
        train_input_ids.append(encoded_sent)


    train_input_ids = pad_sequences(train_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in train_input_ids:
        seq_mask = [float(i>0) for i in seq]
        train_attention_masks.append(seq_mask)

    train_inputs = torch.tensor(train_input_ids, device=args.device)
    train_masks = torch.tensor(train_attention_masks, device=args.device)

    valid_input_ids = []
    valid_attention_masks = []
    for sent in valid_input_sent:
        encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        valid_input_ids.append(encoded_sent)

    valid_input_ids = pad_sequences(valid_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in valid_input_ids:
        seq_mask = [float(i>0) for i in seq]
        valid_attention_masks.append(seq_mask)

    valid_inputs = torch.tensor(valid_input_ids, device=args.device)
    valid_masks = torch.tensor(valid_attention_masks, device=args.device)

    test_input_ids = []
    test_attention_masks = []
    for sent in test_input_sent:
        encoded_sent = tokenizer.encode(
                                sent,                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                max_length = MAX_LEN,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                        )
        test_input_ids.append(encoded_sent)

    test_input_ids = pad_sequences(test_input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    for seq in test_input_ids:
        seq_mask = [float(i>0) for i in seq]
        test_attention_masks.append(seq_mask)

    test_inputs = torch.tensor(test_input_ids, device=args.device)
    test_masks = torch.tensor(test_attention_masks, device=args.device)

    if args.noise_type == "synthetic" and args.syn_type == "IDN":
        del train_noisy_labels, valid_noisy_labels
        train_noisy_labels = corrupt_dataset_IDN(args, train_inputs.cpu(), train_true_labels)
        valid_noisy_labels = corrupt_dataset_IDN(args, valid_inputs.cpu(), valid_true_labels)

    embedding_model = SentenceTransformer(args.embed)
    train_embedding = embedding_model.encode(train_input_sent, convert_to_tensor=True)
    valid_embedding = embedding_model.encode(valid_input_sent, convert_to_tensor=True)
    test_embedding = embedding_model.encode(test_input_sent, convert_to_tensor=True)


    train_data = TensorDataset(train_inputs, train_masks, train_true_labels, train_noisy_labels)
    train_sampler = SequentialSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    valid_data = TensorDataset(valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)

    test_data = TensorDataset(test_inputs, test_masks, test_true_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    return train_data, train_sampler, train_dataloader, train_embedding, valid_data, valid_sampler, valid_dataloader, valid_embedding, test_data, test_sampler, test_dataloader, test_embedding
        
