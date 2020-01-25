import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, 
                 dropout, pad_idx):
        
        super().__init__()
        print("a")
                
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        print("b")
        
        self.convs = nn.ModuleList([nn.Conv2d(in_channels = 1, out_channels = n_filters, kernel_size = (fs, fs)) for fs in filter_sizes])
        print("c")
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        print("d")
        
        self.dropout = nn.Dropout(dropout)
        print("e")
        
    def forward(self, text):
        x = self.embedding(text)
        x = x.squeeze(dim=-1)  
        x = x.unsqueeze(1)  
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool2d(i, i.size(2)).squeeze(2) for i in x]  
        x = t.cat(x, 1)
        x = self.dropout(x)  
        logit = self.fc(x)  
        return logit

        # print("f")
                
        # #text = [batch size, sent len]
        
        # embedded = self.embedding(text)
        # print("g")
                
        # #embedded = [batch size, sent len, emb dim]
        
        # embedded = embedded.unsqueeze(1)
        # print("h")
        
        # #embedded = [batch size, 1, sent len, emb dim]
        
        # conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        # print("i")
            
        # #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        # pooled = [F.max_pool2d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # print(len(pooled))
        # print("j")
        
        # #pooled_n = [batch size, n_filters]
        
        # cat = self.dropout(t.cat(pooled, dim = 1))
        # print(cat)
        # print("k")

        # #cat = [batch size, n_filters * len(filter_sizes)]
            
        # return self.fc(cat)