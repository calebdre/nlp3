import torch
import torch.nn as nn
import torch.nn.functional as F

class GRU(nn.Module):
    def __init__(self, embedding, embedding_size, encoding_size, interaction, dropout):
        super(GRU, self).__init__()

        self.encoding_size = encoding_size
        self.interaction = interaction
        
        self.embedding = nn.Embedding.from_pretrained(embedding)
        self.gru = nn.GRU(embedding_size, encoding_size, 1, batch_first=True, bidirectional=True, dropout=dropout)
        
        inner_lin_dim = encoding_size * 2 * 2 if interaction == "concat" else encoding_size * 2
        self.inner_layer = nn.LSTM(inner_lin_dim, int(encoding_size / 2), 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(encoding_size, 3)
        
    def forward(self, x1, x2):
        batch_size, seq_len = x1.size()
        # encode
        encodeds = []        
        for x in (x1, x2):
            #embed
            embeded = self.embedding(x)
            
            #encode
            encoded, hidden = self.gru(embeded)
            encodeds.append(encoded)

        #interact
        if self.interaction == "concat":
            interacted = self.interaction_concat(*encodeds)
        elif self.interaction == "add":
            interacted = self.interaction_add(*encodeds)
        elif self.interaction == "subtract":
            interacted = self.interaction_subtract(*encodeds)
        elif self.interaction == "mult":
            interacted = self.interaction_multiply(*encodeds)
        else:
            raise Exception("'{}' is an invalid interaction".format(self.interaction))
            
        #fully-connected
        inner = self.inner_layer(interacted)[0]
        inner = torch.sum(inner, dim=1)
        
        classified = self.classifier(inner)
        classified = F.relu(classified.contiguous().view(-1, classified.size(-1))).view(batch_size, classified.size(-1))
        
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

