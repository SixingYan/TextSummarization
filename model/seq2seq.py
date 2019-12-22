r'''
序列模型
使用最简单的
'''
from const import EOS_token, SOS_token

import torch.nn.functional as F
import torch
from torch import nn
from torch import optim

from typing import Dict


# 人为地在句尾加入停止符号

data = [{'X': [2, 3, 4, 5, 7, 5, 0], 'Y':[5, 6, 0]},
        {'X': [2, 3, 4, 5, 7, 5, 0], 'Y':[5, 6, 0]}]
vocab_size = 8
device = 'cpu'
MAX_LENGTH_X = 15
MAX_LENGTH_Y = 5


class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # output_size 类似于词表大小
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # 因为它是一个词一个词地放进去，所以size才是1，同时batch也是1，剩下的应该是hidden_size长度
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        # the output of gru is a tensor of shape (seq_len, batch, num_directions *
        # hidden_size)
        # here is (1,1,)
        output, hidden = self.gru(output, hidden)
        # 所以这里output[0]是 1 by hidden_size 的tensor
        # linear 把 1 by hidden_size 放进去，拿到 1 by outputsize的tensor
        output = self.softmax(self.out(output[0]))  # 为什么这里是0
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH_X):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


teacher_forcing_ratio = 0.5


def train(input_tensor, target_tensor, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion,
          in_maxlen=MAX_LENGTH_X, out_maxlen=MAX_LENGTH_Y):

    # print(input_tensor)
    # print(target_tensor)

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    # print(input_length)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(
        in_maxlen, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    # True if random.random() < teacher_forcing_ratio else False
    use_teacher_forcing = False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        # 这里有个问题，如果decode的句子提前结束，怎么办？如果decode的句子超过了target的长度，怎么算损失？
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(data, encoder, decoder, learning_rate=0.01):

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for i in range(len(data)):
        # 这里的数据输入写法是临时的
        training_pair = data[i]
        input_tensor = torch.tensor(
            training_pair['X'], dtype=torch.long, device=device).view(-1, 1)
        target_tensor = torch.tensor(
            training_pair['Y'], dtype=torch.long, device=device).view(-1, 1)

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print(loss)


def test():
    hidden_size = 256
    encoder1 = EncoderRNN(vocab_size, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(
        hidden_size, vocab_size, dropout_p=0.1).to(device)

    trainIters(data, encoder1, attn_decoder1)


def do():
    """
    向外调用的接口
    """
    pass


if __name__ == '__main__':
    # 本地测试
    test()
