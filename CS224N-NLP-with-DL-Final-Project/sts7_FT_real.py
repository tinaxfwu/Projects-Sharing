import random, numpy as np, argparse
from types import SimpleNamespace
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score

from tokenizer import BertTokenizer
from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm


TQDM_DISABLE=False

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDatasetModified,
    SentencePairTestDatasetModified,
    load_multitask_data
)


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

import spacy

class BertSTS(torch.nn.Module):
    '''
    This module performs sentiment classification using BERT embeddings on the SST dataset.

    In the SST dataset, there are 5 sentiment categories (from 0 - "negative" to 4 - "positive").
    Thus, your forward() should return one logit for each of the 5 classes.
    '''
    def __init__(self, config):
        super(BertSTS, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Pretrain mode does not require updating BERT paramters.
        for param in self.bert.parameters():
            if config.option == 'pretrain':
                param.requires_grad = False
            elif config.option == 'finetune':
                param.requires_grad = True

        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.lstm_cell = torch.nn.LSTM(config.hidden_size, 300, 2, batch_first = True, bidirectional = True) #was 100 at first
        self.config = config


    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2, sent_1, sent_2, device):
        sentence_1 = self.dropout(self.bert(input_ids_1, attention_mask_1)["last_hidden_state"])
        _ , (sentence_1, _) = self.lstm_cell(sentence_1)
        sentence_2 = self.dropout(self.bert(input_ids_2, attention_mask_2)["last_hidden_state"])
        _ , (sentence_2, _) = self.lstm_cell(sentence_2)
        sentence_1 = sentence_1.mean(dim = 0)
        sentence_2 = sentence_2.mean(dim = 0)
        sentence_1 = torch.fft.fft(sentence_1).real
        sentence_2 = torch.fft.fft(sentence_2).real

        return F.cosine_similarity(sentence_1, sentence_2)


# Evaluate the model on dev examples.
def model_eval(dataloader, model, device):
    model.eval() # Switch to eval model, will turn off randomness like dropout.
    sts_y_true = []
    sts_y_pred = []
    sts_sent_ids = []
    with torch.no_grad():
        for step, batch_sts in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
            (b_ids1, b_mask1,
                b_ids2, b_mask2,
            b_labels, b_sent_ids, b_sent1, b_sent2, b_sent_ids) = (batch_sts['token_ids_1'], batch_sts['attention_mask_1'],
                        batch_sts['token_ids_2'], batch_sts['attention_mask_2'],
                        batch_sts['labels'], batch_sts['sent_ids'], batch_sts['sent1'], batch_sts['sent2'], batch_sts['sent_ids'])
                
            b_ids1 = b_ids1.to(device)
            b_mask1 = b_mask1.to(device)
            b_ids2 = b_ids2.to(device)
            b_mask2 = b_mask2.to(device)

            logits = model(b_ids1, b_mask1, b_ids2, b_mask2, b_sent1, b_sent2, device)
            y_hat = logits.flatten().cpu().numpy()
            b_labels = b_labels.flatten().cpu().numpy()

            sts_y_pred.extend(y_hat)
            sts_y_true.extend(b_labels)
            sts_sent_ids.extend(b_sent_ids)
        pearson_mat = np.corrcoef(sts_y_pred,sts_y_true)
        sts_corr = pearson_mat[1][0]


    return sts_corr


# Evaluate the model on test examples.
def model_test_eval(dataloader, model, device):
    model.eval() # Switch to eval model, will turn off randomness like dropout.
    y_pred = []
    sents = []
    sent_ids = []
    for step, batch in enumerate(tqdm(dataloader, desc=f'eval', disable=TQDM_DISABLE)):
        b_ids, b_mask, b_sents, b_sent_ids, sent1, sent2 = batch['token_ids'],batch['attention_mask'],  \
                                                         batch['sents'], batch['sent_ids']
        b_ids = b_ids.to(device)
        b_mask = b_mask.to(device)

        logits = model(b_ids, b_mask)
        logits = logits.detach().cpu().numpy()
        preds = np.argmax(logits, axis=1).flatten()

        y_pred.extend(preds)
        sents.extend(b_sents)
        sent_ids.extend(b_sent_ids)

    return y_pred, sents, sent_ids


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")


def train(args):
    device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
    # Create the data and its corresponding datasets and dataloader.
    train_data = load_data(args.train, 'train')
    dev_data = load_data(args.dev, 'valid')

    #indices = torch.arange(100)
    #train_data = torch.utils.data.Subset(train_data, indices)
    #dev_data = torch.utils.data.Subset(dev_data, indices)

    train_dataset = SentencePairDatasetModified(train_data, args)
    dev_dataset = SentencePairDatasetModified(dev_data, args)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size,
                                collate_fn=dev_dataset.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'hidden_size': 768,
              'data_dir': '.',
              'option': args.option}

    config = SimpleNamespace(**config)

    model = BertSTS(config)
    model = model.to(device)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    # Run for the specified number of epochs.
    # print(train_dataloader)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0
        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            token_1, attention_mask_1, token_2, attention_mask_2, labels, sent1, sent2 = (batch['token_ids_1'], batch['attention_mask_1'], batch['token_ids_2'], batch['attention_mask_2'], batch['labels'], batch['sent1'], batch['sent2'])
            token_1 = token_1.to(device)
            attention_mask_1 = attention_mask_1.to(device)
            token_2 = token_2.to(device)
            attention_mask_2 = attention_mask_2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(token_1, attention_mask_1, token_2, attention_mask_2, sent1, sent2, device)
            #logits = model(token_1, attention_mask_1, token_2, attention_mask_2, sent1, sent2, device)
            #loss = F.cross_entropy(logits, labels.view(-1).float(), reduction='sum') / (args.batch_size)
            #loss = logits.mean()
            #print(loss)
            #sentence_1 = sentence_1.to(device)
            #sentence_2 = sentence_2.to(device)
            #obj = torch.tensor(1)
            #obj = obj.to(device)
            #lossfn = torch.nn.CosineEmbeddingLoss()
            #loss = lossfn(logits, labels.view(-1).float(),obj)
            logits = logits * 5
            loss = F.mse_loss(logits, labels.view(-1).float())
            # print(loss)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / (num_batches)

        train_acc = model_eval(train_dataloader, model, device)
        dev_acc = model_eval(dev_dataloader, model, device)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, train acc :: {train_acc :.3f}, dev acc :: {dev_acc :.3f}")


def test(args):
    with torch.no_grad():
        device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
        saved = torch.load(args.filepath)
        config = saved['model_config']
        model = BertSTS(config)
        model.load_state_dict(saved['model'])
        model = model.to(device)
        print(f"load model from {args.filepath}")
        
        dev_data = load_data(args.dev, 'valid')
        dev_dataset = SentencePairDatasetModified(dev_data, args)
        dev_dataloader = DataLoader(dev_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=dev_dataset.collate_fn)

        test_data = load_data(args.test, 'test')
        test_dataset = SentencePairTestDatasetModified(test_data, args)
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size, collate_fn=test_dataset.collate_fn)
        
        dev_acc, dev_f1, dev_pred, dev_true, dev_sents, dev_sent_ids = model_eval(dev_dataloader, model, device)
        print('DONE DEV')
        test_pred, test_sents, test_sent_ids = model_test_eval(test_dataloader, model, device)
        print('DONE Test')
        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.dev_out, "w+") as f:
            print(f"dev acc :: {dev_acc :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sent_ids,dev_pred ):
                f.write(f"{p} , {s} \n")

        with open(args.test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s  in zip(test_sent_ids,test_pred ):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--option", type=str,
                        help='pretrain: the BERT parameters are frozen; finetune: BERT parameters are updated',
                        choices=('pretrain', 'finetune'), default="pretrain")
    parser.add_argument("--use_gpu", action='store_true')

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate, default lr for 'pretrain': 1e-3, 'finetune': 1e-5",
                        default=1e-3)

    args = parser.parse_args()
    return args

def preprocess_string(s):
    return ' '.join(s.lower()
                    .replace('.', ' .')
                    .replace('?', ' ?')
                    .replace(',', ' ,')
                    .replace('\'', ' \'')
                    .split())


def load_data(similarity_filename,split='train'):
    similarity_data = []
    if split == 'test':
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2'])
                                        ,sent_id))
    else:
        with open(similarity_filename, 'r') as fp:
            for record in csv.DictReader(fp,delimiter = '\t'):
                sent_id = record['id'].lower().strip()
                similarity_data.append((preprocess_string(record['sentence1']),
                                        preprocess_string(record['sentence2']),
                                        float(record['similarity']),sent_id))

    print(f"Loaded {len(similarity_data)} {split} examples from {similarity_filename}")

    return similarity_data



if __name__ == "__main__":
    args = get_args()
    seed_everything(args.seed)

    print('Training STS Classifier on Quora...')
    config = SimpleNamespace(
        filepath='sts-classifier-mod.pt',
        lr=args.lr,
        use_gpu=args.use_gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dropout_prob=args.hidden_dropout_prob,
        train='data/sts-train.csv',
        dev='data/sts-dev.csv',
        test='data/sts-test-student.csv',
        option=args.option,
        dev_out = 'predictions/' + args.option + '-sts-single-dev-out.csv',
        test_out = 'predictions/' + args.option + '-sts-single-test-out.csv'
    )

    train(config)

    #print('Evaluating on Quora...')
    #test(config)
