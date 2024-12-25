import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel
from re_implement.crf import CRF


class BiRnnCrf(nn.Module):
    def __init__(self, tagset_size, hidden_dim, num_rnn_layers=1, rnn="lstm"):
        super(BiRnnCrf, self).__init__()
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size

        # Sử dụng mô hình PhoBERT làm pretrained embedding
        self.embedding_model = AutoModel.from_pretrained("vinai/phobert-base")
        for param in self.embedding_model.parameters():
            param.requires_grad = False

        embedding_dim = self.embedding_model.config.hidden_size

        # RNN layer (LSTM hoặc GRU)
        RNN = nn.LSTM if rnn == "lstm" else nn.GRU
        self.rnn = RNN(embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                       bidirectional=True, batch_first=True)
        
        # CRF layer
        self.crf = CRF(hidden_dim, self.tagset_size)

    def __build_features(self, input_ids, attention_masks):
        """
        sentences: tokenized input IDs
        attention_masks: attention masks corresponding to the input IDs
        """
        # Tạo embedding từ PhoBERT
        with torch.no_grad():
            outputs = self.embedding_model(input_ids=input_ids, attention_mask=attention_masks)
            embeds = outputs.last_hidden_state  # Lấy hidden state của layer cuối cùng

        seq_length = attention_masks.sum(1)  # Độ dài thực tế của từng câu (không tính padding)
        
        # Chuyển sang CPU để sắp xếp
        seq_length_cpu = seq_length.cpu()
        sorted_seq_length, perm_idx = seq_length_cpu.sort(descending=True)
        
        # Chuyển perm_idx về cùng device với embeds để index
        perm_idx = perm_idx.to(embeds.device)
        embeds = embeds[perm_idx, :]

        # Sử dụng sorted_seq_length trên CPU cho pack_padded_sequence
        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Khôi phục thứ tự ban đầu
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]
        attention_masks = attention_masks[unperm_idx, :]

        return lstm_out, attention_masks
    
    def loss(self, input_ids, attention_masks, tags):
        features, masks = self.__build_features(input_ids, attention_masks)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, input_ids, attention_masks):
        """
        xs: tokenized input IDs
        attention_masks: attention masks
        """
        features, masks = self.__build_features(input_ids, attention_masks)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
