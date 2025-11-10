import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
nltk.download('stopwords')
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
from torch.utils.data import TensorDataset, DataLoader
import sacrebleu
from sacrebleu.metrics import BLEU
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)


train_src=[]
train_targ=[]
test_src=[]
test_targ=[]
dev_src=[]
dev_targ=[]

def load_sentences(file_path):
	sentences = []
	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			sentence = line.strip()
			sentences.append(sentence)
	return sentences

def preprocess_en(sentences):
	preprocessed_sentences = []
	stop_words = set(stopwords.words('english'))
	punctuation = set(string.punctuation)
	for sentence in sentences:
		words = word_tokenize(sentence)
		filtered_words = ["<SOS>"]
		for word in words:
			if word.lower() not in stop_words and word not in punctuation:
				filtered_words.append(word.lower())
		filtered_words.append("<EOS>")
		if len(filtered_words) < 128:
			filtered_words += ['<PAD>'] * (128 - len(filtered_words))
		preprocessed_sentences.append(filtered_words[:128])
	return preprocessed_sentences

def preprocess_fr(sentences):
	preprocessed_sentences = []
	stop_words = set(stopwords.words('french'))
	punctuation = set(string.punctuation)
	for sentence in sentences:
		words = word_tokenize(sentence)
		filtered_words = ["<SOS>"]
		for word in words:
			if word.lower() not in stop_words and word not in punctuation:
				filtered_words.append(word.lower())
		filtered_words.append("<EOS>")
		if len(filtered_words) < 128:
			filtered_words += ['<PAD>'] * (128 - len(filtered_words))
		preprocessed_sentences.append(filtered_words[:128])
	return preprocessed_sentences


train_src=load_sentences("./ted-talks-corpus/train.en")
test_src=load_sentences("./ted-talks-corpus/test.en")
dev_src=load_sentences("./ted-talks-corpus/dev.en")
train_targ=load_sentences("./ted-talks-corpus/train.fr")
test_targ=load_sentences("./ted-talks-corpus/test.fr")
dev_targ=load_sentences("./ted-talks-corpus/dev.fr")

train_src=preprocess_en(train_src)
test_src=preprocess_en(test_src)
dev_src=preprocess_en(dev_src)
train_targ=preprocess_fr(train_targ)
test_targ=preprocess_fr(test_targ)
dev_targ=preprocess_fr(dev_targ)

source_vocab={}
target_vocab={}
target_id_to_word=[]

for i in [train_src,dev_src,test_src]:
	for j in i:
		for k in j:
			if k not in source_vocab:
				source_vocab[k]=len(source_vocab)
for i in [train_targ,dev_targ,test_targ]:
	for j in i:
		for k in j:
			if k not in target_vocab:
				target_vocab[k]=len(target_vocab)

for i in target_vocab:
	target_id_to_word.append(i)

def sentences_to_tensors_src(sentences, vocab):
	tensor_data = []
	for sentence in sentences:
		ids = [vocab.get(token, vocab['<PAD>']) for token in sentence]
		tensor_data.append(ids)
	tensor_data = torch.tensor(tensor_data)
	return tensor_data

def sentences_to_tensors_targ(sentences, vocab):
	tensor_data = []
	for sentence in sentences:
		ids = []
		for token in sentence:
			if token == '<EOS>':
				ids.append(vocab.get('<PAD>'))
				continue
			ids.append(vocab.get(token, vocab['<PAD>']) )
		ids.append(vocab.get('<PAD>'))
		tensor_data.append(ids)
	tensor_data = torch.tensor(tensor_data)
	return tensor_data

train_src_tensor = sentences_to_tensors_src(train_src, source_vocab)
test_src_tensor = sentences_to_tensors_src(test_src, source_vocab)
dev_src_tensor = sentences_to_tensors_src(dev_src, source_vocab)
train_tgt_tensor = sentences_to_tensors_targ(train_targ, target_vocab)
test_tgt_tensor = sentences_to_tensors_targ(test_targ, target_vocab)
dev_tgt_tensor = sentences_to_tensors_targ(dev_targ, target_vocab)


train_dataset = TensorDataset(train_src_tensor, train_tgt_tensor)
test_dataset = TensorDataset(test_src_tensor, test_tgt_tensor)
dev_dataset = TensorDataset(dev_src_tensor, dev_tgt_tensor)

## Transformers Architecture

class multiheadedattention(nn.Module):
	def __init__(self, model_dim, headsct):
		super(multiheadedattention,self).__init__()
		self.model_dim=model_dim
		self.headsct=headsct
		self.key_dim=model_dim//headsct

		self.query=nn.Linear(model_dim, model_dim)
		self.value=nn.Linear(model_dim, model_dim)
		self.key=nn.Linear(model_dim, model_dim)
		self.output_layer=nn.Linear(model_dim, model_dim)

	def split_heads(self, inp):
		batch, seq_len, model_dim=inp.size()
		split=inp.view(batch, seq_len, self.headsct, self.key_dim)
		split=split.transpose(1,2)
		return split

	def combine_heads(self, inp):
		batch, _, seq_len, key_dim=inp.size()
		combine=inp.transpose(1,2).contiguous()
		combine=combine.view(batch, seq_len, self.model_dim)
		return combine

	def calculate(self, q, k, v, mask):
		k=k.transpose(-2,-1)
		scores=torch.matmul(q,k)/math.sqrt(self.key_dim)
		if mask != None:
			scores=scores.masked_fill(mask==0,-1e9)
		probabilities=torch.softmax(scores,dim=-1)
		output=torch.matmul(probabilities,v)
		return output

	def forward(self, Query, Key, Value, mask=None):
		Q=self.query(Query)
		q=self.split_heads(Q)
		K=self.key(Key)
		k=self.split_heads(K)
		V=self.value(Value)
		v=self.split_heads(V)  

		attention_out=self.calculate(q, k, v, mask)

		output=self.output_layer(self.combine_heads(attention_out))

		return output

class feedforward(nn.Module):
	def __init__(self, model_dim, feedforward_dim):
		super(feedforward, self).__init__()
		self.layer1=nn.Linear(model_dim, feedforward_dim)
		self.layer2=nn.Linear(feedforward_dim, model_dim)
		self.relu=nn.ReLU()

	def forward(self, inp):
		out1=self.layer1(inp)
		out_relu=self.relu(out1)
		out2=self.layer2(out_relu)
		return out2

class positionalenc(nn.Module):
	def __init__(self, model_dim, max_seq_len):
		super(positionalenc, self).__init__()

		pos_enc=torch.zeros(max_seq_len, model_dim)
		pos=torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
		scaler=torch.exp(torch.arange(0, model_dim, 2).float() * -(math.log(10000.0)/model_dim))

		pos_enc[:,1::2]=torch.cos(pos*scaler)
		pos_enc[:,0::2]=torch.sin(pos*scaler)

		self.register_buffer("pos_enc",pos_enc.unsqueeze(0))

	def forward(self, inp):
		enc=self.pos_enc[:,:inp.size(1)]
		return inp+enc


class encoder(nn.Module):
	def __init__(self, model_dim, headsct, feedforward_dim, dropout):
		super(encoder, self).__init__()
		self.attention=multiheadedattention(model_dim, headsct)
		self.feed_forward=feedforward(model_dim, feedforward_dim)
		self.normalize1=nn.LayerNorm(model_dim)
		self.normalize2=nn.LayerNorm(model_dim)
		self.dropout=nn.Dropout(dropout)

	def forward(self, inp, mask):
		attention_out=self.attention(inp, inp, inp, mask)
		attention_dropout=self.dropout(attention_out)
		inp=self.normalize1(inp+attention_dropout)
		feedforward_out=self.feed_forward(inp)
		feedforward_dropout=self.dropout(feedforward_out)
		output=self.normalize2(inp+feedforward_dropout)
		return output

class decoder(nn.Module):
	def __init__(self, model_dim, headsct, feedforward_dim, dropout):
		super(decoder, self).__init__()
		self.attention=multiheadedattention(model_dim, headsct)
		self.crossattention=multiheadedattention(model_dim, headsct)
		self.feed_forward=feedforward(model_dim, feedforward_dim)
		self.normalize1=nn.LayerNorm(model_dim)
		self.normalize2=nn.LayerNorm(model_dim)
		self.normalize3=nn.LayerNorm(model_dim)
		self.dropout=nn.Dropout(dropout)

	def forward(self, inp, encoder_output, source_mask, target_mask):
		attention_out=self.attention(inp, inp, inp, target_mask)
		attention_dropout=self.dropout(attention_out)
		inp=self.normalize1(inp+attention_dropout)
		attention_out=self.crossattention(inp, encoder_output, encoder_output, source_mask)
		attention_dropout=self.dropout(attention_out)
		inp=self.normalize2(inp+attention_dropout)
		feedforward_out=self.feed_forward(inp)
		feedforward_dropout=self.dropout(feedforward_out)
		output=self.normalize3(inp+feedforward_dropout)
		return output


class transformer(nn.Module):
	def __init__(self, source_vocab_size, target_vocab_size, model_dim, headsct, feedforward_dim, max_seq_len, dropout):
		super(transformer, self).__init__()
		self.max_len=max_seq_length
		self.enc_embedds=nn.Embedding(source_vocab_size, model_dim)
		self.dec_embedds=nn.Embedding(target_vocab_size, model_dim)
		self.positional_enc=positionalenc(model_dim, max_seq_len)
		self.enc_layer1=encoder(model_dim, headsct, feedforward_dim, dropout)
		self.enc_layer2=encoder(model_dim, headsct, feedforward_dim, dropout)
		self.dec_layer1=decoder(model_dim, headsct, feedforward_dim, dropout)
		self.dec_layer2=decoder(model_dim, headsct, feedforward_dim, dropout)
		self.lin=nn.Linear(model_dim, target_vocab_size)
		self.dropout=nn.Dropout(dropout)

	def masking(self, source, target):
		source_mask=(source!=0).unsqueeze(1).unsqueeze(2)
		source_mask=source_mask.to(device)
		target_mask=(target!=0).unsqueeze(1).unsqueeze(3)
		target_mask=target_mask.to(device)
		nopeak_mask=(1-torch.triu(torch.ones(1, self.max_len, self.max_len), diagonal=1)).bool()
		nopeak_mask=nopeak_mask.to(device)
		target_mask=target_mask & nopeak_mask
		return source_mask, target_mask

	def forward(self, source, target):
		source_mask, target_mask = self.masking(source, target)
		source_enc_emb=self.enc_embedds(source)
		target_enc_emb=self.dec_embedds(target)
		source_pos=self.positional_enc(source_enc_emb)
		target_pos=self.positional_enc(target_enc_emb)
		source_embedds=self.dropout(source_pos)
		target_embedds=self.dropout(target_pos)
		encoder_out1=self.enc_layer1(source_embedds, source_mask)
		encoder_out2=self.enc_layer2(encoder_out1, source_mask)
		decoder_out1=self.dec_layer1(target_embedds, encoder_out2, source_mask, target_mask)
		decoder_out2=self.dec_layer2(decoder_out1, encoder_out2, source_mask, target_mask)
		output=self.lin(decoder_out2)
		return output

batch_size = 64

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size)

source_vocab_size = len(source_vocab)
target_vocab_size = len(target_vocab)
model_dim = 512
num_heads = 16
feedforward_dim = 2048
max_seq_length = 128
dropout = 0.1
lr=0.00001
transformer_model = transformer(source_vocab_size, target_vocab_size, model_dim, num_heads, feedforward_dim, max_seq_length, dropout)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

num_epochs = 10
transformer_model.to(device)
val_losses=[]
train_losses=[]
scores=[]
result=[]
for epoch in range(num_epochs):
	transformer_model.train()
	total_loss = 0.0
	for source, target in tqdm(train_dataloader):
		source=source.to(device)
		target=target.to(device)
		optimizer.zero_grad()
		output=transformer_model(source, target[:,:-1])
		modified_target=target.clone()
		indices=(modified_target==8).nonzero()
		for i in range(len(indices)):
			row,col=indices[i]
			if modified_target[row,col]==8:
				modified_target[row,col]=7
				break
		output=output.reshape(-1,target_vocab_size)
		modified_target=modified_target[:,1:].reshape(-1)
		loss=criterion(output, modified_target)
		loss.backward()
		optimizer.step()
		total_loss+=loss.item()
	avg_train=total_loss/len(train_dataloader)
	train_losses.append(avg_train)
	transformer_model.eval()
	total_loss=0.0
	references = []
	hypotheses = []
	with torch.no_grad():
		for source, target in dev_dataloader:
			source=source.to(device)
			target=target.to(device)
			output=transformer_model(source, target[:,:-1])
			modified_target=target.clone()
			indices=(modified_target==8).nonzero()
			for i in range(len(indices)):
				row,col=indices[i]
				if modified_target[row,col]==8:
					modified_target[row,col]=7
					break
			output_=output.reshape(-1,target_vocab_size)
			modified_target=modified_target[:,1:].reshape(-1)
			loss=criterion(output_, modified_target)
			total_loss+=loss.item()
			_,predicted=torch.max(output,dim=-1)
			for ref, hyp in zip(target[:,:-1].cpu().numpy(), predicted.cpu().numpy()):
				ref_sent=[]
				hyp_sent=[]
				for i in range(1,len(ref)):
					if ref[i] == 8:
						break
					ref_sent.append(target_id_to_word[ref[i]])
				for i in range(1,len(hyp)):
					if hyp[i] == 8:
						break
					hyp_sent.append(target_id_to_word[hyp[i]])
				references.append(' '.join(ref_sent))
				hypotheses.append(' '.join(hyp_sent))
	avg_dev=total_loss/len(dev_dataloader)
	bleu=BLEU()
	bleu_score = bleu.corpus_score(hypotheses, references)
	print("Epoch: ",epoch+1, "Train Loss: ", avg_train, end=" ")
	print("Validation Loss: ", avg_dev, "Validation score: ", bleu_score.score)
	val_losses.append(avg_dev)
	scores.append(bleu_score.score)

torch.save(transformer_model.state_dict(), 'transformer_model.pth')

# Loading the saved model

model = transformer(source_vocab_size, target_vocab_size, model_dim, num_heads, feedforward_dim, max_seq_length, dropout)

model.load_state_dict(torch.load('transformer_model.pth', map_location=torch.device(device)))

model.to(device)

model.eval()

with torch.no_grad():
	for source, target in train_dataloader:
		source=source.to(device)
		target=target.to(device)
		output=model(source, target[:,:-1])
		_,predicted=torch.max(output,dim=-1)
		bleu=BLEU()
		for ref, hyp in zip(target[:,:-1].cpu().numpy(), predicted.cpu().numpy()):
			ref_sent=[]
			hyp_sent=[]
			for i in range(1,len(ref)):
				if ref[i] == 8:
					break
				ref_sent.append(target_id_to_word[ref[i]])
			for i in range(1,len(hyp)):
				if hyp[i] == 8:
					break
				hyp_sent.append(target_id_to_word[hyp[i]])
			true=' '.join(ref_sent)
			pred=' '.join(hyp_sent)
			bleu_score = sacrebleu.sentence_bleu(pred, [true])
			print(pred,'\t',true,'\t',bleu_score.score)



