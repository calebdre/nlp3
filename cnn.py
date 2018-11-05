import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding, kernel_size, embedding_size, hidden_size, interaction, dropout):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.interaction = interaction
        self.dropout = dropout
        self.padding = 1
        self.embedding = nn.Embedding.from_pretrained(embedding)

        self.conv1 = nn.Conv1d(embedding_size, hidden_size, kernel_size=kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=kernel_size, padding=self.padding)
        
        self.max_pool = nn.MaxPool1d(hidden_size)
                
        inner_lin_dim = hidden_size * 2 if interaction == "concat" else hidden_size
        self.classifier = nn.Linear(hidden_size , 3)
        
        if interaction == "concat":
            self.interaction = self.interaction_concat
        elif interaction == "add":
            self.interaction = self.interaction_add
        elif interaction == "subtract":
            self.interaction = self.interaction_subtract
        else:
            raise Exception("'{}' is an invalid interaction".format(self.interaction))
        
    def calc_conv_dim1_size(self, seq_len):
        return seq_len + ((self.padding*2) - (self.kernel_size - 1))
    
    def conv_dim1_sizes(self, seq_len):
        conv1_size = self.calc_conv_dim1_size(seq_len)
        conv2_size = self.calc_conv_dim1_size(conv1_size)
        output_size = self.calc_conv_dim1_size(conv2_size)
        return conv1_size, conv2_size, output_size
    
    def forward(self, x1, x2):
        batch_size, seq_len = x1.size()
        conv1_dim1_size, conv2_dim1_size, output_size = self.conv_dim1_sizes(seq_len)

        sent1_embed = self.embedding(x1)
        sent2_embed = self.embedding(x2)
        encodeds = []
        for embed in (sent1_embed, sent2_embed):
            hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
            contig = hidden.contiguous().view(-1, hidden.size(-1))
            hidden = F.relu(contig).view(batch_size, conv1_dim1_size, hidden.size(-1))
            
            hidden = nn.Dropout(self.dropout)(hidden)
            
            hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
            hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, conv2_dim1_size, hidden.size(-1))
            
            hidden = nn.Dropout(self.dropout)(hidden)
            
            hidden = nn.MaxPool1d(hidden.shape[1])(hidden.transpose(1, 2)).squeeze()
            encodeds.append(hidden)
        
        interacted = self.interaction(*encodeds)
        
        lstm = nn.LSTM(1, self.hidden_size, dropout=self.dropout, num_layers=2, bidirectional=False, batch_first=True)
        l1, _ = lstm(interacted.view(batch_size, interacted.shape[-1], -1))
        
#         l1 = nn.MaxPool1d(l1.shape[-1])(l1).squeeze()
        l1 = torch.sum(l1, dim=1)
        l1 = nn.Linear(
            interacted.shape[-1], 
            int(interacted.shape[-1] / 2)
        )(interacted)
        l1 = F.softmax(l1, -1)
            
        classified = nn.Linear(l1.shape[-1], 3)(l1)        
        classified = F.softmax(classified, -1)
        return classified
            
    def interaction_concat(self, sent1, sent2):
        return torch.cat((sent1, sent2), 1) 
    
    def interaction_add(self, sent1, sent2):
        return sent1 - sent2
    
    def interaction_subtract(self, sent1, sent2):
        return torch.mul(sent1, sent2)
    
    

