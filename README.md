# pytorch-NER
RoBERTa + BiLSTM + CRF for Chinese NER Task

## Requirement
- python 3.8
- pytorch 1.8.1
- transformers 4.11
- tqdm

## CLUENER
fine-grained named entity recognition dataset and benchmark for chinese [[see also]](clue/README.md)

## zh-RoBERTa
use the pretrained RoBERTa weight for chinese from @brightmart and @ymcui [[see also]](resource/README.md)

## Run

```shell
python main.py
```

### configure what you want

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # device
lr = 1e-3  # learning_rate
batch_size = 128  # batch size
accumulation_steps = 1  # accumulation steps for gradient accumulation
roberta_path = 'resource/RoBERTa_zh_Large_PyTorch'  # path to PLM files

lstm_hidden_dim = 768  # hidden dim for BiLSTM
lstm_dropout_rate = 0.2  # dropout rate for BiLSTM
```


## Result

|Entity|Precision|Recall|F1|
|---|---|---|---|
|address|0.6171|0.6005|0.6087|
|book|0.8322|0.7727|0.8013|
|company|0.8306|0.8042|0.8172|
|game|0.7578|0.9017|0.8235|
|government|0.8154|0.8583|0.8363|
|movie|0.8862|0.7219|0.7956|
|name|0.8750|0.8882|0.8815|
|organization|0.7506|0.8104|0.7794|
|position|0.7924|0.7667|0.7793|
|scene|0.7129|0.7129|0.7129|
|**macro**|0.7819|0.7895|0.7857|
|**micro**|0.7870|0.7838|0.7836|



