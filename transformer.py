import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset
import math
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import string
from torchtext.vocab import build_vocab_from_iterator
import re
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

punc_str = string.punctuation+'\n'
#load data
train_df = pd.read_csv(r'E:\pycharm\PycharmProjects\nlp_tweets_disaster\train.csv')
sent_list = train_df['text'].str.lower().tolist()
#regrex pattern
comp = re.compile('[^A-Z^a-z^0-9^ ]')
http_reg = re.compile('http[s]?[^\s]*\s*')
uni_reg = re.compile(r'\\x[a-f0-9]{2}')
num_reg = re.compile('[0-9]')
multi_reg = re.compile(r'(.)\1{3,}')
vocabulary_list = []
sent_len_list = []
# lowercase
for i in range(len(sent_list)):
    #removing punctuation
    tmp_str = sent_list[i].translate(str.maketrans('','',punc_str))
    #English
    tmp_str = comp.sub('', tmp_str)
    # drop http
    tmp_str = http_reg.sub('',tmp_str)
    # drop unicode
    tmp_str = uni_reg.sub('',tmp_str)
    #drop numbers
    tmp_str = num_reg.sub('',tmp_str)
    #drop duplicates
    tmp2_str = multi_reg.sub(r'\1',tmp_str)

    tmp_list = sent_tokenize(tmp2_str)[0].split(' ')
    sent_list[i] = tmp_list
    sent_len_list.append(len(tmp_list))
    vocabulary_list+=tmp_list
    #stemming
    '''stemmer = nltk.stem.porter.PorterStemmer()'''
    #lemmatization

###################################################### this setting is very important!!!!!!!!! ##################################################################
#id representation
PAD_IDX = 2
special_symbols = ['<negative>','<positive>','<pad>','<unk>']
#the sequence of special_symbol is very important as in the end, 0 denotes negative, 1 denotes positive, instead of any other words. so you have to specify 0,1 are neg,pos sepearately
vocabulary_list = list(set(vocabulary_list))
vocabulary_list.remove('')

id_transform = build_vocab_from_iterator(sent_list,min_freq=1,specials=special_symbols,special_first=True)
id_transform.set_default_index(3)
#label list
label_list = train_df.loc[:,'target'].tolist()


class sent_dataset(Dataset):
    def __init__(self,word_list,label_list):
        super().__init__()
        self.word_list = word_list
        self.label_list = label_list
    def __getitem__(self,index):
        return self.word_list[index],self.label_list[index]

    def __len__(self):
        return len(self.word_list)

####################################   mask #########################################################
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def create_src_mask(src):
    src_seq_len = src.shape[0]

    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    return src_mask, src_padding_mask
def collate_fn(batch_data):
    word_list,label_list = zip(*batch_data)
    enc_list = []
    tgt_list = []
    for idx in range(len(word_list)):
        enc_list.append(torch.tensor(id_transform(word_list[idx])))
        tgt_list.append(torch.tensor([label_list[idx]]))
    enc_padded = pad_sequence(enc_list,padding_value=PAD_IDX)
    tgt_padded = pad_sequence(tgt_list,padding_value=PAD_IDX)
    return enc_padded,tgt_padded
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self,vocab_size_int,emb_size_int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size_int,emb_size_int)
        self.emb_size_int = emb_size_int

    def forward(self,token_tensor):
        return self.embedding(token_tensor.long())*np.sqrt(self.emb_size_int) # times sqrt to rescale embeddings in order to adjust to positional encoding
class ClassifyTransformer(nn.Module):
    def __init__(self,num_encoder_layers_int,
                 emb_size_int,nhead_int,src_vocab_size_int,tgt_vocab_size_int,dim_feedforward_int=512,
                 dropout_float=0.1):
        super(ClassifyTransformer,self).__init__()
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=emb_size_int, nhead = nhead_int)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer,num_layers=num_encoder_layers_int)
        self.generator = nn.Linear(emb_size_int,tgt_vocab_size_int)
        self.src_tok_emb = TokenEmbedding(src_vocab_size_int,emb_size_int)
        self.positional_encoding = PositionalEncoding(emb_size_int,dropout_float)
    def forward(self,src_seq,src_mask,src_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src_seq))
        outs = self.transformer_encoder(src_emb,mask = src_mask,src_key_padding_mask=src_padding_mask)
        outs = self.generator(outs)
        #avg pooling
        outs = torch.mean(outs,dim = 0)
        return outs

def train_loop(model,optimizer,input_dataset,loss_func):
    data_loader = DataLoader(input_dataset,batch_size=batch_size_int,collate_fn=collate_fn,shuffle=True)
    model.train()
    loss_list = []
    for src,tgt in data_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask,src_padding_mask = create_src_mask(src)
        src_mask = src_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        logits = model(src,src_mask,src_padding_mask)
        optimizer.zero_grad()
        loss = loss_func(logits.view(-1,logits.shape[-1]),tgt.view(-1))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            loss_list.append(loss.item())
        del src
        del tgt
        del logits
    print(f'AVG Loss: {np.mean(loss_list)}')
def val_loop(model,input_dataset):
    data_loader = DataLoader(input_dataset,batch_size=len(input_dataset),collate_fn=collate_fn,shuffle=True)
    model.eval()
    for src,tgt in data_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask, src_padding_mask = create_src_mask(src)
        src_mask = src_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        pred = model(src, src_mask, src_padding_mask)
        pred = pred.cpu().detach().numpy()
        pred_label_array = pred.argmax(axis=1)
        tgt = tgt.cpu().detach().numpy()[0]
        correct_pred_int = np.sum(pred_label_array==tgt)
        accuracy = np.round(correct_pred_int/len(tgt)*100,2)
        print(f'val set Accuracy: {accuracy}%')
        return pred_label_array
        del src
        del tgt
        del pred

batch_size_int = 1024
epochs = 60
lr = 5e-3
emb_size_int = 512
src_vocab_size_int = len(id_transform)
dec_vocab_size_int = 2
ffn_hide_dim_int = 512
num_encoder_layers_int = 12
nhead_int = 4
word_dataset = sent_dataset(sent_list,label_list)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# model instance
transformer = ClassifyTransformer(num_encoder_layers_int, emb_size_int,
                                 nhead_int, src_vocab_size_int, dec_vocab_size_int)
for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)


optimizer = torch.optim.SGD(transformer.parameters(),lr=lr)

#dataset split
dataset_len_int = len(word_dataset)
val_size_int = int(0.1*dataset_len_int)
train_size_int = dataset_len_int-val_size_int
train_dataset,val_dataset = train_test_split(word_dataset,test_size=val_size_int,train_size=train_size_int)

pred_df = pd.DataFrame(columns = list(range(val_size_int)))
for epoch in range(1, epochs+1):
    print(f'epoch: {epoch}')
    train_loop(transformer, optimizer,train_dataset,loss_fn)
    sub_pred_array = val_loop(transformer, val_dataset)
    pred_df.loc[epoch,:] = sub_pred_array.reshape(-1,)
#forecasting

test_df = pd.read_csv('test.csv')
sent_list = test_df['text'].str.lower().tolist()
for i in range(len(sent_list)):
    #removing punctuation
    tmp_str = sent_list[i].translate(str.maketrans('','',punc_str))
    # drop http
    tmp_str = http_reg.sub('',tmp_str)
    # drop unicode
    tmp_str = uni_reg.sub('',tmp_str)
    #drop numbers
    tmp_str = num_reg.sub('',tmp_str)
    #drop duplicates
    tmp2_str = multi_reg.sub(r'\1',tmp_str)

    tmp_list = sent_tokenize(tmp2_str)[0].split(' ')
    sent_list[i] = tmp_list

word_dataset = sent_dataset(sent_list,label_list)
word_loader = DataLoader(word_dataset,batch_size=1,collate_fn=collate_fn)
model = transformer
model.eval()
for src,tgt in word_loader:
    src = src.to(DEVICE)
    src_mask, src_padding_mask = create_src_mask(src)
    src_mask = src_mask.to(DEVICE)
    src_padding_mask = src_padding_mask.to(DEVICE)
    pred = model(src, src_mask, src_padding_mask)
    pred = pred.detach().numpy()[-1]
    pred_label_array = pred.argmax(axis=1)
submission_df = pd.read_csv('sample_submission.csv')
submission_df['target'] = pred_label_array






