import torch.nn as nn
import torch.nn.functional as F
import torch


class CNN(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=64, n_filters=[128, 256, 512], filter_sizes=[3, 3, 3], pad_idx=0):
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

    def forward(self, text):
        embedded = self.embedding(text)
        embedded = embedded.unsqueeze(1)

        conved_0 = F.relu(self.conv_0(embedded).squeeze(3))
        conved_1 = F.relu(self.conv_1(embedded).squeeze(3))
        conved_2 = F.relu(self.conv_2(embedded).squeeze(3))

        pooled_0 = F.avg_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.avg_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.avg_pool1d(conved_2, conved_2.shape[2]).squeeze(2)

        cat = torch.cat((pooled_0, pooled_1, pooled_2), dim=1)

        # TODO, create Bi-GRU to get the sentence representations

        return cat


embeddings = torch.randint(low=1, high=400, size=(10, 50)).long()

coherence = CNN()

coherence(embeddings)
