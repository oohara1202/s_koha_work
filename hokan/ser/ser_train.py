import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchaudio
import torchinfo

import utils
from mel_processing import mel_spectrogram_torch

# BATCH FIRST TimeDistributed layer
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # squash samples and timesteps into a single axis
        elif len(x.size()) == 3:  # (samples, timesteps, inp1)
            x_reshape = x.contiguous().view(-1, x.size(2))  # (samples * timesteps, inp1)
        elif len(x.size()) == 4:  # (samples, timesteps, inp1, inp2)
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3))  # (samples*timesteps, inp1, inp2)
        else:  # (samples, timesteps, inp1, inp2, inp3)
            x_reshape = x.contiguous().view(-1, x.size(2), x.size(3), x.size(4))  # (samples*timesteps, inp1, inp2, inp3)

        y = self.module(x_reshape)

        # we have to reshape Y
        if len(x.size()) == 3:
            y = y.contiguous().view(x.size(0), -1, y.size(1))  # (samples, timesteps, out1)
        elif len(x.size()) == 4:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2))  # (samples, timesteps, out1, out2)
        else:
            y = y.contiguous().view(x.size(0), -1, y.size(1), y.size(2),
                                    y.size(3))  # (samples, timesteps, out1, out2, out3)
        
        return y

class HybridModel(nn.Module):
    def __init__(self,num_emotions):
        super().__init__()
        # conv block
        self.conv2Dblock = nn.Sequential(
            # 1. conv block
            TimeDistributed(nn.Conv2d(in_channels=1,
                                   out_channels=16,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(16)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=2, stride=2)),
            TimeDistributed(nn.Dropout(p=0.3)),
            # 2. conv block
            TimeDistributed(nn.Conv2d(in_channels=16,
                                   out_channels=32,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(32)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3)),
            # 3. conv block
            TimeDistributed(nn.Conv2d(in_channels=32,
                                   out_channels=64,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1
                                  )),
            TimeDistributed(nn.BatchNorm2d(64)),
            TimeDistributed(nn.ReLU()),
            TimeDistributed(nn.MaxPool2d(kernel_size=4, stride=4)),
            TimeDistributed(nn.Dropout(p=0.3))
        )

        # LSTM block
        hidden_size = 32
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, bidirectional=True, batch_first=True)
        self.dropout_lstm = nn.Dropout(p=0.4)
        self.attention_linear = nn.Linear(2*hidden_size, 1) # 2*hidden_size for the 2 outputs of bidir LSTM
        
        # Linear softmax layer
        self.out_linear = nn.Linear(2*hidden_size, num_emotions)

    def forward(self, x):
        conv_embedding = self.conv2Dblock(x)
        conv_embedding = torch.flatten(conv_embedding, start_dim=2) # do not flatten batch dimension and time
        lstm_embedding, (h, c) = self.lstm(conv_embedding)
        lstm_embedding = self.dropout_lstm(lstm_embedding)
        # lstm_embedding (batch, time, hidden_size*2)
        batch_size, T, _ = lstm_embedding.shape
        attention_weights = [None]*T
        for t in range(T):
            embedding = lstm_embedding[: ,t, :]
            attention_weights[t] = self.attention_linear(embedding)
        attention_weights_norm = nn.functional.softmax(torch.stack(attention_weights, -1), dim=-1)
        attention = torch.bmm(attention_weights_norm, lstm_embedding) # (Bx1xT)*(B,T,hidden_size*2)=(B,1,2*hidden_size)
        attention = torch.squeeze(attention, 1)
        output_logits = self.out_linear(attention)
        output_softmax = nn.functional.softmax(output_logits, dim=1)
        
        return output_logits, output_softmax

def loss_fnc(predictions, targets):
    return nn.CrossEntropyLoss()(input=predictions, target=targets)

def make_train_step(model, loss_fnc, optimizer):
    def train_step(X, Y):
        # set model to train mode
        model.train()
        # forward pass
        output_logits, output_softmax = model(X)
        predictions = torch.argmax(output_softmax, dim=1)
        accuracy = torch.sum(Y==predictions)/float(len(Y))
        # compute loss
        loss = loss_fnc(output_logits, Y)
        # compute gradients
        loss.backward()
        # update parameters and zero gradients
        optimizer.step()
        optimizer.zero_grad()
        return loss.item(), accuracy*100
    return train_step

def make_validate_fnc(model, loss_fnc):
    def validate(X, Y):
        with torch.no_grad():
            model.eval()
            output_logits, output_softmax = model(X)
            predictions = torch.argmax(output_softmax, dim=1)
            accuracy = torch.sum(Y==predictions)/float(len(Y))
            loss = loss_fnc(output_logits, Y)
        return loss.item(), accuracy*100, predictions
    return validate

class AudioCollate():
    """ Zero-pads model inputs
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from mel-spectrogram
        PARAMS
        ------
        batch:[t, c ,h ,w]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[0].shape[0] for x in batch]),
            dim=0, descending=True)

        max_chunk_len = max([x[0].shape[0] for x in batch])

        chunk_padded = torch.FloatTensor(len(batch),
                                         max_chunk_len,
                                         batch[0][0].shape[1],
                                         batch[0][0].shape[2],
                                         batch[0][0].shape[3])
        label_padded = torch.LongTensor(len(batch))

        chunk_padded.zero_()
        label_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            chunk = row[0]
            chunk = torch.from_numpy(chunk).clone()
            chunk_padded[i, :chunk.size(0), :, :, :] = chunk

            label = row[1]
            label_padded[i] = label

        if self.return_ids:
            return chunk_padded, label_padded, ids_sorted_decreasing
        return chunk_padded, label_padded

class AudioEmotionLoader(torch.utils.data.Dataset):
    def __init__(self, filelists_file, hps):
        self.mel_chunked = list()  # チャンク化されたメルスペクトログラム
        self.labels = list()       # ラベル

        with open(filelists_file, mode='r', encoding='utf-8') as tr_f:
            line = [s.rstrip().split('|') for s in tr_f]
            filepath_list, label_list = [s[0] for s in line], [int(s[1]) for s in line]

        for wav_file in filepath_list:
            audio, _ = torchaudio.load(wav_file)
            mel_spec = mel_spectrogram_torch(
                audio, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )
            mel_spec = mel_spec.squeeze(0).numpy()  # 元のプログラムに合わせるためにnumpy変換
            chunks = self._splitIntoChunks(mel_spec, win_size=hps.data.chunk_win_size, stride=hps.data.chunk_stride)
            chunks = np.expand_dims(chunks, 1)  # (t, h, w) --. (t, c, h, w)
            self.mel_chunked.append(chunks)

        self.labels = label_list.copy()

    def _splitIntoChunks(self, mel_spec, win_size, stride):
        """
        PARAMS
        ------
        win_size: chunkの大きさ
        stride:  chunkが移動する幅
        """
        t = mel_spec.shape[1]  # メルスペクトログラムのフレーム数
        num_of_chunks = int(t/stride)
        chunks = list()
        for i in range(num_of_chunks):
            chunk = mel_spec[:, i*stride:i*stride+win_size]
            if chunk.shape[1] == win_size:
                chunks.append(chunk)
            # ファイル終端のchunkに対してzero padding
            else:
                chunk = np.pad(chunk, ((0, 0), (0, win_size-chunk.shape[1])), 'constant')
                chunks.append(chunk)
        return np.stack(chunks, axis=0)

    def __len__(self):
        return len(self.mel_chunked)
        
    def __getitem__(self, idx):
        return self.mel_chunked[idx], self.labels[idx]

def main():
    EMOTIONS = {0:'Neutral', 1:'Happy', 2:'Sad', 3:'Angry'}
    # EMOTIONS = {0:'Neutral', 1:'Happy', 2:'Sad'}
    
    hps = utils.get_hparams()

    print("Loading train data")
    train_data = AudioEmotionLoader(hps.data.training_files, hps)
    print("Loading test data")
    test_data = AudioEmotionLoader(hps.data.test_files, hps)

    collate_fn = AudioCollate()

    train_loader = DataLoader(train_data,
                              batch_size=hps.train.batch_size,
                              num_workers=hps.train.num_workers,
                              shuffle=True,
                              collate_fn=collate_fn
                              )
    test_loader = DataLoader(test_data,
                             batch_size=hps.train.batch_size,
                             num_workers=hps.train.num_workers,
                             shuffle=False,
                             collate_fn=collate_fn
                             )
    
    train_data_size = len(train_data)
    test_data_size = len(test_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Selected device is {}'.format(device))
    
    model = HybridModel(num_emotions=len(EMOTIONS)).to(device)
    torchinfo.summary(model,
                      input_size=(hps.train.batch_size, 10, 1, hps.data.n_mel_channels, hps.data.chunk_win_size)
                      )
    print('Number of trainable params: ', sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=hps.train.learning_rate,
                                weight_decay=hps.train.weight_decay,
                                momentum=hps.train.momentum
                                )

    train_step = make_train_step(model, loss_fnc, optimizer=optimizer)
    validate = make_validate_fnc(model, loss_fnc)
    losses=[]
    val_losses = []
    iters = int(train_data_size / hps.train.batch_size)
    for epoch in range(hps.train.epochs):
        epoch_acc = 0
        epoch_loss = 0

        for i, data in enumerate(train_loader):
            X, Y = data
            X_tensor = X.clone().detach().to(device).float()
            Y_tensor = Y.clone().detach().to(device).long()
            loss, acc = train_step(X_tensor, Y_tensor)
            epoch_acc += acc*hps.train.batch_size/train_data_size
            epoch_loss += loss*hps.train.batch_size/train_data_size
            print(f"\r Epoch {epoch}: iteration {i}/{iters}", end='')
        
        epoch_val_acc = 0
        epoch_val_loss = 0
        for X, Y in test_loader:
            X_val_tensor = X.clone().detach().to(device).float()
            Y_val_tensor = Y.clone().detach().to(device).long()
            val_loss, val_acc, predictions = validate(X_val_tensor, Y_val_tensor)
            epoch_val_acc += val_acc*hps.train.batch_size/test_data_size
            epoch_val_loss += val_loss*hps.train.batch_size/test_data_size
        losses.append(epoch_loss)
        val_losses.append(epoch_val_loss)
        # tb.add_scalar("Training Loss", epoch_loss, epoch)
        # tb.add_scalar("Training Accuracy", epoch_acc, epoch)
        # tb.add_scalar("Validation Loss", val_loss, epoch)
        # tb.add_scalar("Validation Accuracy", val_acc, epoch)
        print('')
        print(f"Epoch {epoch} --> loss:{epoch_loss:.4f}, acc:{epoch_acc:.2f}%, val_loss:{epoch_val_loss:.4f}, val_acc:{epoch_val_acc:.2f}%", flush=True)

if __name__ == "__main__":
  main()
