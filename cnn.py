import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, embedding, kernel_size, embedding_size, hidden_size, interaction, dropout, lengths):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.interaction_t = interaction
        self.dropout = nn.Dropout(dropout)
        self.padding = 1
        
        self.embedding = nn.Embedding.from_pretrained(embedding)

        h2 = int(hidden_size / 2)
        h4 = int(hidden_size / 4)
    
        self.conv1 = nn.Conv1d(embedding_size, hidden_size, kernel_size=kernel_size, padding=self.padding)
        k2 = 1 if kernel_size <= 2 else kernel_size - 2
        self.conv2 = nn.Conv1d(hidden_size, h2, kernel_size=k2, padding=self.padding)
        self.max_pool = nn.MaxPool1d(h2)
        
        self.setup_inner_layer(lengths, h4)
        self.classifier = nn.Linear(h4, 3)
    
        self.set_interaction(interaction)
        
    def forward(self, x1, x2):
        batch_size, seq_len = x1.size()
        conv1_dim1_size, conv2_dim1_size, output_size = self.conv_dim1_sizes(seq_len)

        sent1_embed = self.embedding(x1)
        sent2_embed = self.embedding(x2)
        encodeds = []
        for embed in (sent1_embed, sent2_embed):
            hidden = self.conv1(embed.transpose(1,2)).transpose(1,2)
            hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, -1, hidden.size(-1))
                        
            hidden = self.conv2(hidden.transpose(1,2)).transpose(1,2)
            hidden = F.relu(hidden.contiguous().view(-1, hidden.size(-1))).view(batch_size, -1, hidden.size(-1))

            hidden = self.max_pool(hidden).squeeze()
            hidden = self.dropout(hidden)
            
            encodeds.append(hidden)
        
        interacted = self.interaction(*encodeds)
        l1 = self.inner_layer(seq_len, interacted)
        l1 = F.relu(l1.contiguous().view(-1, l1.size(-1))).view(batch_size, l1.size(-1))
        classified = self.classifier(l1)        
        classified = F.softmax(classified, -1)
        return classified
    
    def setup_inner_layer(self, lengths, output_size):
        layers = {}
        for l in lengths:
            if l not in layers:
                s1, s2, out = self.conv_dim1_sizes(l)
                if s2 < 1:
                    s2 = 1
                s2 = s2 * 2 if self.interaction_t == "concat" else s2
                layers[l] = nn.Linear(s2, output_size)
        
        self.inner_layer_defs = layers

    def inner_layer(self, l, x):
        if torch.cuda.is_available():
            return self.inner_layer_defs[l].cuda()(x)
        else:
            return self.inner_layer_defs[l](x)
    
    def set_interaction(self, interaction):
        if interaction == "concat":
            self.interaction = self.interaction_concat
        elif interaction == "add":
            self.interaction = self.interaction_add
        elif interaction == "sub":
            self.interaction = self.interaction_subtract
        elif interaction == "mult":
            self.interaction = self.interaction_mult
        else:
            raise Exception("'{}' is an invalid interaction".format(self.interaction))
            
    def interaction_concat(self, sent1, sent2):
        return torch.cat((sent1, sent2), 1) 
    
    def interaction_add(self, sent1, sent2):
        return sent1 + sent2
    
    def interaction_subtract(self, sent1, sent2):
        return sent1 - sent2
    
    def interaction_mult(self, sent1, sent2):
        return sent1 * sent2
    
    def calc_conv_dim1_size(self, seq_len, kernel_size):
        return seq_len + ((self.padding*2) - (kernel_size - 1))
    
    def conv_dim1_sizes(self, seq_len):
        conv1_size = self.calc_conv_dim1_size(seq_len, self.kernel_size)
        conv2_size = self.calc_conv_dim1_size(conv1_size, self.kernel_size - 2 if self.kernel_size > 2 else 1)
        output_size = self.calc_conv_dim1_size(conv2_size, self.kernel_size)
        return conv1_size, conv2_size, output_size