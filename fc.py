import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
from datasets import load_dataset
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self, vocab_size, hidden_size, n_layers, dropout, embed_size):
        super().__init__()

        self.input_size = vocab_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embed_size = embed_size
        self.embedding = nn.Embedding(self.input_size, self.embed_size)

        self.embedding2hidden = nn.Linear(embed_size*1000, hidden_size)
        self.hidden2label = nn.Linear(hidden_size, 6)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropoutLayer = nn.Dropout(0.3)

    def forward(self, inputs):
        output = self.embedding(inputs) 

        output = output.permute(0, 2, 1)
        output = output.reshape(len(output), -1)

        output = self.embedding2hidden(output)
        output = F.relu(output)
        output = self.dropoutLayer(output)
        output = self.hidden2label(output)

        output = self.softmax(output)
        return output
    

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
    # text_tensor = text_tensor.permute(1,0)

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

    model = Net(vocab_size=len(charlist), 
                        hidden_size=512, 
                        n_layers=5, 
                        dropout=0.3,
                        embed_size=64)
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    steps  = int(len(ds["train"])/batch_size*epochs)
    print_step = steps/10

    for step in tqdm(range(steps)):
        model.train()
        model.cuda()
        model.zero_grad()

        input_tensor, input_lengths, target_tensor = create_batch(list(ds["train"]), charlist, batch_size)

        input_tensor = input_tensor.reshape(batch_size, -1).cuda()

        output = model.forward(input_tensor)
        output = output.cpu()

        # print(f"{output.shape=}")
        loss = criterion(output, target_tensor)
        loss.backward()
        optimizer.step()

        if step % print_step:
            continue

        model.eval()

        correct = 0

        input_tensor, input_lengths, target_tensor = create_batch(list(ds["validation"]), charlist, len(ds["validation"]))
        input_tensor = input_tensor.cuda()

        output = model(inputs=input_tensor,)
        preds = torch.argmax(output.cpu().detach(), dim=1)


        for i in range(len(preds)):
            if preds[i] == target_tensor[i]:
                correct += 1

        accuracy = correct/len(preds)

        print(f"{loss=}")
        print(f"{accuracy=}")

main()
