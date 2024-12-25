from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm

dataset = load_dataset('leduckhai/VietMed-NER')
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def data_download():
    # Tải dữ liệu dưới dạng file .json
    if not os.path.exists('vietmed_data'):
        os.makedirs('vietmed_data')
    dataset['train'].to_json('vietmed_data/train.json')
    dataset['validation'].to_json('vietmed_data/val.json')
    dataset['test'].to_json('vietmed_data/test.json')

# Tạo label_tag dictionary
def create_label_tag_dict(train_data):
    label_tag_dict = {}
    for tags, labels in zip (train_data['tags'], train_data['labels']):
        for tag, label in zip (tags, labels):
            if label not in label_tag_dict:
                label_tag_dict[label] = tag
    sorted_dict = dict(sorted(label_tag_dict.items(), key=lambda item: item[1]))
    with open('vietmed_data/labels_tags.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_dict, f, ensure_ascii=False, indent=4)
    return sorted_dict



# Đổi format của data thành ["word1", "word2", "word3"]     ["label1", "label2", "label3"]
def changing_data_format(dataset):
    split_names = dataset.keys()
    for split_name in split_names:
        file_path_name = f'vietmed_data/{split_name}.txt'
        lines = []
        for sample in dataset[split_name]:
            sentence = sample['words']
            labels = sample['labels']
            lines.append(f"{sentence}\t{labels}\n")
        with open(file_path_name, 'w', encoding='utf-8') as f:
            f.writelines(lines)

# Tokenize ['word1', 'word2', 'word3']
def sentence_to_inputids_and_mask(sentence):
    encode = tokenizer(sentence, 
                       padding='max_length', 
                       is_split_into_words=True, # Xử lý câu ở dạng ['word1', 'word2', 'word3']
                       max_length=60, 
                       truncation=True, # Cắt câu khi vượt max_length
                       return_tensors='pt')
    input_ids = encode['input_ids']
    attention_mask = encode['attention_mask']
    # token = tokenizer.convert_ids_to_tokens(input_ids) # input_ids to token
    return input_ids, attention_mask


def labels_to_labelsids(input_ids, labels):
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    special_token = ['<s>', '</s>', '<pad>']
    labelids = []
    i = 0
    previous_is_subword = False
    for token in tokens:
        if token in special_token:
            labelids.append(-100)
        elif token[-1] == '@':
            if not previous_is_subword:
                labelids.append(labels[i])
                previous_is_subword = True
            else:
                label = 'I-' + labels[i].split('-')[1] if labels[i] != '0' else '0'
                labelids.append(label)
        else:
            if previous_is_subword:
                label = 'I-' + labels[i].split('-')[1] if labels[i] != '0' else '0'
                labelids.append(label)
                previous_is_subword = False
            else:
                labelids.append(labels[i])
            i += 1
    return labelids

def labelids_to_tags(labelids, dict):
    tags = []
    for label in labelids:
        tags.append(dict.get(label, -100)) # tag cua cac token <s>, </s>, <pad> la -100
    return tags

def load_data(data_path):
    input_ids_data = []
    attention_mask_data = []
    tags_data = []

    with open(data_path, 'r', encoding='utf-8') as f:
        data = f.read()

    with open('vietmed_data/labels_tags.json', 'r') as file:
        my_dict = json.load(file)

    for sample in tqdm(data.split('\n')[:-1]):
        
        sentence, labels = sample.split('\t')
        sentence = sentence[2:-2].split("', '")
        labels = labels[2:-2].split("', '")

        input_ids, attention_mask = sentence_to_inputids_and_mask(sentence)
        tags = labelids_to_tags(labels_to_labelsids(input_ids[0], labels), dict=my_dict)

        input_ids_data.append(input_ids.squeeze())
        attention_mask_data.append(attention_mask.squeeze())
        tags_data.append(tags)
    print("Load data successfully!")

    return input_ids_data, attention_mask_data, tags_data


# Chuan bi data cho model

class NERDataset(Dataset):
    def __init__(self, input_ids, attention_mask, tags):
        """
        Khởi tạo Dataset cho NER.
        Args:
            input_ids (List[List[int]]): Danh sách token IDs cho mỗi câu.
            attention_mask (List[List[int]]): Danh sách attention masks cho mỗi câu.
            tags (List[List[int]]): Danh sách nhãn cho mỗi token.
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tags = tags
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.input_ids[idx], dtype=torch.long),
            'attention_mask': torch.tensor(self.attention_mask[idx], dtype=torch.long),
            'tags': torch.tensor(self.tags[idx], dtype=torch.long)
        }
    
### Kiem tra qua trinh chuan bi du lieu truoc khi train model
input_ids, attention_mask, tags = load_data('vietmed_data/train.txt')
# Khởi tạo Dataset
dataset = NERDataset(input_ids, attention_mask, tags)

# Khởi tạo DataLoader
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

for batch in data_loader:
    input_ids = batch['input_ids']  # [batch_size, seq_length]
    attention_mask = batch['attention_mask']  # [batch_size, seq_length]
    tags = batch['tags']  # [batch_size, seq_length]
    
    print(input_ids.shape, attention_mask.shape, tags.shape)
    break
    
            
if __name__ == "__main__":
    from transformers import AutoModel
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    with open('vietmed_data/labels_tags.json', 'r') as file:
        my_dict = json.load(file)
    # Tokenize văn bản
    sentence = ['thì', 'cũng', 'giống', 'như', 'ba', 'má', 'mình']
    encoding = tokenizer(sentence, 
                        padding='max_length',
                        is_split_into_words=True,
                        truncation=True, 
                        max_length=15,
                        return_tensors='pt')

    print(encoding['input_ids'])
    print(encoding['attention_mask'])
    input_ids = encoding['input_ids'][0]
    labels = ['0', '0', '0', '0', 'B-GENDER', 'B-GENDER', '0']
    labelids = labels_to_labelsids(input_ids, labels)
    print(labelids)
    tags = labelids_to_tags(labelids, my_dict)
    print(tags)
    # with open('vietmed_data/train.txt', 'r', encoding='utf-8') as f:
    #     train_data = f.read()
    # input_ids_data, attention_mask_data, tags_data = load_data(train_data)
    # print(len(input_ids_data), len(attention_mask_data), len(tags_data))

