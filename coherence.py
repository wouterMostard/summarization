import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=64, n_filters=[128, 256, 256], filter_sizes=[3, 5, 7], pad_idx=0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        self.conv_0 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters[0],
                                kernel_size=(filter_sizes[0], embedding_dim))

        self.conv_1 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters[1],
                                kernel_size=(filter_sizes[1], embedding_dim))

        self.conv_2 = nn.Conv2d(in_channels=1,
                                out_channels=n_filters[2],
                                kernel_size=(filter_sizes[2], embedding_dim))

        self.gru = nn.GRU(input_size=640, bidirectional=True, hidden_size=256, batch_first=False)

        self.doc_mlp = nn.Sequential(
            nn.Linear(512, 256),  # TODO, determine the right output shape
            nn.Tanh()
        )

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)
        print(embedded.shape)

        conved_0 = self.conv_0(embedded).squeeze(3)
        print(conved_0.shape)
        conved_1 = self.conv_1(embedded).squeeze(3)
        conved_2 = self.conv_2(embedded).squeeze(3)

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        print(pooled_0.shape)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = torch.cat((pooled_0, pooled_1, pooled_2), dim=1)
        print(cat.shape)

        x = cat.mean(dim=0, keepdims=True)

        x = torch.unsqueeze(torch.vstack([x, x, x, x]), 0) # TODO, so now we just have four of the same sentences in 1 document
        print(x.shape)
        output, hidden = self.gru(x)
        print(hidden.shape)

        h = torch.cat([hidden[0], hidden[1]], dim=1)

        print(h.shape)

        mean_sent = h.mean(dim=0, keepdims=True)
        print(mean_sent.shape)

        d = self.doc_mlp(mean_sent)
        print(d.shape)

        return x


embeddings = torch.randint(low=1, high=400, size=(10, 50)).long()

coherence = CNN()

coherence(embeddings)
