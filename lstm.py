import random

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import functional as F
from torch.autograd import Variable
from datasets import load_dataset
from tqdm import tqdm
from matplotlib import pyplot as plt


class RNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout, embed_size):
        super().__init__()

        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.rnn = nn.LSTM(input_size=self.embed_size,
                           hidden_size=hidden_size,
                           dropout=dropout,
                           num_layers=n_layers, bidirectional=True)
        self.hidden2label = nn.Linear(2*hidden_size, 6)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropoutLayer = nn.Dropout()

    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_size).cuda())
        return h0, c0

    def forward(self, inputs, input_lengths): 
        self.hidden = self.init_hidden(inputs.size(-1)) 
        embedded = self.embedding(inputs) 
        packed = pack_padded_sequence(embedded, input_lengths, batch_first=False) 
        outputs, self.hidden = self.rnn(packed, self.hidden)
        output, output_lengths = pad_packed_sequence(outputs, batch_first=False)

        output = torch.transpose(output, 0, 1)

        output = torch.transpose(output, 1, 2)

        output = torch.tanh(output)

        output, indices = F.max_pool1d(output,output.size(2), return_indices=True)

        output = torch.tanh(output)
        output = output.squeeze(2)
        output = self.dropoutLayer(output)

        output = self.hidden2label(output)

        output = self.softmax(output)
        return output, self.hidden
    

def create_batch(data, charlist, batch_size=1, maxchar=1000):
    indices = random.sample(range(len(data)), batch_size)

    batch = [data[index] for index in indices]
    vectorized_texts = [[charlist.index(token) for token in elem["text"]] for elem in batch]
    text_tensor = Variable(torch.zeros((len(vectorized_texts), maxchar))).long()
    text_lengths = [len(elem["text"]) for elem in batch]

    for idx, (text, textlen) in enumerate(zip(vectorized_texts, text_lengths)):
        text_tensor[idx, :textlen] = torch.LongTensor(text)

    sorting = sorted([(e, i) for i, e in enumerate(text_lengths)], reverse=True)
    perm_idx = [x[1] for x in sorting]
    text_lengths = [x[0] for x in sorting]
    text_tensor = text_tensor[perm_idx]
    text_tensor = text_tensor.permute(1,0)

    labels = [elem["label"] for elem in batch]
    label_tensor = torch.LongTensor(batch_size)
    for i, label in enumerate(labels):
        label_tensor[i] = label
    
    label_tensor = label_tensor[perm_idx]

    return text_tensor, text_lengths, label_tensor


def main():
    epochs = 20
    batch_size = 200

    ds = load_dataset("dair-ai/emotion", "split")
    
    charlist = ""
    for text in ds["train"]:
        charlist += text["text"]

    charlist = sorted(set(charlist))

    model = RNN(vocab_size=len(charlist), 
                        hidden_size=64, 
                        n_layers=2, 
                        dropout=0.3,
                        embed_size=64)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    steps  = int(len(ds["train"])/batch_size*epochs)
    print_step = steps/10

    train_step_x = []
    loss_y = []
    val_step_x = []
    val_accs_y = []
    for step in tqdm(range(steps)):
        train_step_x.append(step)

        model.train()
        model.hidden = model.init_hidden(batch_size)
        model.cuda()
        model.zero_grad()

        input_tensor, input_lengths, target_tensor = create_batch(list(ds["train"]), charlist, batch_size)

        input_tensor = input_tensor.cuda()

        output, _ = model.forward(input_tensor, input_lengths)
        output = output.cpu()
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        loss_y.append(loss.detach().numpy())

        if step % print_step:
            continue

        model.eval()

        val_step_x.append(step)

        correct = 0

        input_tensor, input_lengths, target_tensor = create_batch(list(ds["validation"]), charlist, len(ds["validation"]))
        input_tensor = input_tensor.cuda()

        output, hidden = model(inputs = input_tensor, input_lengths = input_lengths)
        preds = torch.argmax(output.cpu().detach(), dim=1)


        for i in range(len(preds)):
            if preds[i] == target_tensor[i]:
                correct += 1

        accuracy = correct/len(preds)
        val_accs_y.append(accuracy)

        print(f"{loss=}")
        print(f"{accuracy=}")

    plt.plot(train_step_x, loss_y, label="loss")
    # plt.plot(val_step_x, val_accs_y, label="accuracy")
    plt.legend()
    plt.savefig("./assets/lstm_hidden_size_64.jpg")

main()
