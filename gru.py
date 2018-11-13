import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, embedding, embedding_size, encoding_size, interaction, dropout):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.gru = nn.GRU(embedding_size, encoding_size, 1, batch_first=True, bidirectional=True)
        
        inner_lin_dim = encoding_size * 2 if interaction == "concat" else encoding_size
        lstm_out = int(encoding_size / 2)
        self.inner_layer = nn.LSTM(inner_lin_dim, lstm_out, 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(lstm_out* 2, 3)
        self.dropout = nn.Dropout(dropout)
        
        if interaction == "concat":
            self.interaction = self.interaction_concat
        elif interaction == "add":
            self.interaction = self.interaction_add
        elif interaction == "sub":
            self.interaction = self.interaction_subtract
        elif interaction == "mult":
            self.interaction = self.interaction_multiply
        else:
            raise Exception("'{}' is an invalid interaction".format(self.interaction))
        
    def forward(self, x1, x2):
        batch_size, seq_len = x1.size()
        # encode
        encodeds = []        
        for x in (x1, x2):
            #embed
            embeded = self.embedding(x)
            
            #encode
            encoded, hidden = self.gru(embeded)
            hidden = hidden.transpose(0,1)
            encodeds.append(hidden)

        #interact
        interacted = self.interaction(*encodeds)
        
        #fully-connected
        inner = self.inner_layer(interacted)[0]
        inner = torch.sum(inner, dim=1)
        
        inner = self.dropout(inner)
        
        classified = self.classifier(inner)
        logits = F.softmax(classified, -1)
        return logits
    
    def interaction_concat(self, x1, x2):
        return torch.cat((x1, x2), 2)
    
    def interaction_add(self, x1, x2):
        return x1 + x2
    
    def interaction_subtract(self, x1, x2):
        return x1 - x2
   
    def interaction_multiply(self, x1, x2):
        return x1 * x2