import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from torch.utils.data.dataset import Dataset
import math
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
import pandas as pd
import string
from torchtext.vocab import build_vocab_from_iterator
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from torch.utils.data import Subset
#nltk.download('stopwords')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lemmatizer = WordNetLemmatizer()
punc_str = string.punctuation+'\n'
#load data
train_df = pd.read_csv(r'E:\pycharm\PycharmProjects\nlp_tweets_disaster\train.csv')

#label list
label_list = train_df.loc[:,'target'].tolist()

sent_list = train_df['text'].str.lower().tolist()
#regrex pattern
comp = re.compile('[^A-Z^a-z^0-9^ ]')
http_reg = re.compile('http[s]?[^\s]*\s*')
uni_reg = re.compile(r'\\x[a-f0-9]{2}')
num_reg = re.compile('[0-9]')
multi_reg = re.compile(r'(.)\1{3,}')


def text_process(sent_list):
    vocabulary_list = []
    # stop words
    stopwords = set(nltk.corpus.stopwords.words('english'))
    stopwords.update(['u', 'im', 'like', 'get', 'dont', '',"may",'us','all'])
    #lemmatizer
    lemmatizer = WordNetLemmatizer()
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
        tmp_str = multi_reg.sub(r'\1',tmp_str)
        tmp_list = sent_tokenize(tmp_str)[0].split(' ')
        #word lemmatizer and remove stop words
        tmp2_list = []
        for word in tmp_list:
            if word not in stopwords:
                tmp2_list.append(lemmatizer.lemmatize(word))
        vocabulary_list+=tmp2_list
        sent_list[i] = tmp2_list
    return sent_list,vocabulary_list
#text process
sent_list,vocabulary_list = text_process(sent_list)


disaster_idx_list = [idx for idx,val in enumerate(label_list) if val >0]
disaster_sent_list = [sent_list[idx] for idx in disaster_idx_list]


#rare words detect
vocab_freq = nltk.FreqDist(sum(disaster_sent_list,[]))
high_freq_list = sorted(vocab_freq.items(),key=lambda x:x[1])
freq_df = pd.DataFrame(high_freq_list)
rare_word_list = freq_df.where(freq_df[1]<=1).dropna(axis=0)[0].tolist()








###################################################### this setting is very important!!!!!!!!! ##################################################################
#id representation
PAD_IDX = 2
special_symbols = ['<negative>','<positive>','<pad>','<unk>']
#the sequence of special_symbol is very important as in the end, 0 denotes negative, 1 denotes positive, instead of any other words. so you have to specify 0,1 are neg,pos sepearately
vocabulary_list = list(set(vocabulary_list))


id_transform = build_vocab_from_iterator(sent_list,min_freq=1,specials=special_symbols,special_first=True)
id_transform.set_default_index(3)


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
        self.layernorm = torch.nn.LayerNorm(emb_size_int)
        self.transformer_encoder = torch.nn.TransformerEncoder(encoder_layer=self.encoder_layer,norm = self.layernorm,num_layers=num_encoder_layers_int)
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

def train_loop(model,optimizer,data_loader,loss_func):
    model.train()
    loss_list = []
    accuracy_list = []
    for idx,(src,tgt) in enumerate(data_loader):
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
            logits = logits.cpu().detach().numpy()
            logits_label_array = logits.argmax(axis=1)
            tgt = tgt.cpu().detach().numpy()[0]
            correct_pred_int = np.sum(logits_label_array == tgt)
            accuracy = np.round(correct_pred_int / len(tgt) * 100, 2)
            accuracy_list.append(accuracy)
        del src
        del tgt
        del logits
    with torch.no_grad():
        loss_mean_float = np.mean(loss_list)
        accuracy_mean_float = np.mean(accuracy_list)
        print(f'AVG Loss: {loss_mean_float}')
        print(f'AVG Accuracy: {np.round(accuracy_mean_float, 2)}%')
        return loss_mean_float, accuracy_mean_float / 100
def val_loop(model,data_loader,loss_func):
    model.eval()
    for src, tgt in data_loader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        src_mask, src_padding_mask = create_src_mask(src)
        src_mask = src_mask.to(DEVICE)
        src_padding_mask = src_padding_mask.to(DEVICE)
        pred = model(src, src_mask, src_padding_mask)
        with torch.no_grad():
            loss = loss_func(pred.view(-1, pred.shape[-1]), tgt.view(-1))
            loss_float = loss.item()
        pred = pred.cpu().detach().numpy()
        pred_label_array = pred.argmax(axis=1)
        tgt = tgt.cpu().detach().numpy()[0]
        correct_pred_int = np.sum(pred_label_array == tgt)
        accuracy = np.round(correct_pred_int / len(tgt) * 100, 2)
        print(f'val set Loss: {loss}')
        print(f'val set Accuracy: {accuracy}%')
        del src
        del tgt
        del pred
        del loss
    return loss_float, accuracy / 100


batch_size_int = 16 # the model can only learn from small batch size(<=48) baing on SGD, otherwise train loss will be stuck at 0.69 whihc means the model doesn't learn.
epochs = 15
lr = 5e-3
emb_size_int = 512
src_vocab_size_int = len(id_transform)
dec_vocab_size_int = 2
#small model seems to give better performance and faster in training
#large model(more layers more nhead) wont learn given high drop rates(>0.4)
num_encoder_layers_int = 2
nhead_int = 4
word_dataset = sent_dataset(sent_list,label_list)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# model instance
transformer = ClassifyTransformer(num_encoder_layers_int, emb_size_int,
                                 nhead_int, src_vocab_size_int, dec_vocab_size_int,)
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
train_dataloader = DataLoader(train_dataset)
val_dataloader = DataLoader(val_dataset)
train_loss_list = []
train_accuracy_list = []
val_loss_list = []
val_accuracy_list = []

for epoch in range(1, epochs+1):
    print(f'epoch: {epoch}')
    train_loss,train_accuracy = train_loop(transformer, optimizer,train_dataloader,loss_fn)
    val_loss,val_accuracy = val_loop(transformer, val_dataset,loss_fn)
    train_loss_list.append(train_loss)
    train_accuracy_list.append(train_accuracy)
    val_loss_list.append(val_loss)
    val_accuracy_list.append(val_accuracy)


#forecasting

test_df = pd.read_csv('test.csv')
sent_list = test_df['text'].str.lower().tolist()
sent_list,vocabulary_list = text_process(sent_list)

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






