import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import spacy
from datasets import Dataset
import torchtext
import tqdm
import evaluate
import os.path as pt
import os
import pandas as pd
from config import Config
from morpheme_data import MorphemeDataLoader

# CHANGES FOR MORPHS

lang = 'hun'

# SOS_TOKEN = "<sos>"
# EOS_TOKEN = "<eos>"
#
# def get_path(dset):
#     data_dir = '../2022SegmentationST/data'
#     return pt.join(pt.dirname(pt.abspath(__file__)), data_dir, f'{lang}.word.{dset}.tsv')
#
#
# def get_data(path):
#     if os.path.exists(path):
#         table = pd.read_table(path, sep='\t', header=None)
#         table.columns = ['words', 'morph_str', 'identifier']
#
#         # print("Trimming to first 10000 lines")
#         # table = table[:1000]
#
#         # Replace ' @@' morpheme marker with '@' which will make things easier
#         # table['morph_str'] = table['morph_str'].str.replace(' @@', MORPH_SEPARATOR)
#         words = [[SOS_TOKEN] + list(w) + [EOS_TOKEN] for w in table['words'].to_numpy()]
#         morphs = [[SOS_TOKEN] + list(m) + [EOS_TOKEN] for m in table['morph_str'].to_numpy()]
#         return {"words": words, "morphs": morphs}
#     else:
#         print("data not found")
#         exit(1)
#
# train_data = get_data(get_path('train'))
# valid_data = get_data(get_path('dev'))
# test_data = get_data(get_path('test.gold'))
#
# ############################
#
# dataset = {'train': train_data,
#         'validation': valid_data,
#         'test': test_data}
#
#
# train_data, valid_data, test_data = (
#     dataset["train"],
#     dataset["validation"],
#     dataset["test"],
# )
#
#
# min_freq = 2
# unk_token = "<unk>"
# pad_token = "<pad>"
#
# special_tokens = [
#     unk_token,
#     pad_token,
#     SOS_TOKEN,
#     EOS_TOKEN,
# ]
#
# word_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["words"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )
#
# morph_vocab = torchtext.vocab.build_vocab_from_iterator(
#     train_data["morphs"],
#     min_freq=min_freq,
#     specials=special_tokens,
# )
#
# assert word_vocab[unk_token] == morph_vocab[unk_token]
# assert word_vocab[pad_token] == morph_vocab[pad_token]
#
# unk_index = word_vocab[unk_token]
# pad_index = word_vocab[pad_token]
#
# word_vocab.set_default_index(unk_index)
# morph_vocab.set_default_index(unk_index)
# fn_kwargs = {"en_vocab": word_vocab, "de_vocab": morph_vocab}
#
# train_data['word_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in train_data['words']]
# train_data['morph_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in train_data['morphs']]
# valid_data['word_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in valid_data['words']]
# valid_data['morph_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in valid_data['morphs']]
# test_data['word_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in test_data['words']]
# test_data['morph_ids'] = [torch.Tensor(word_vocab.lookup_indices(char_list)) for char_list in test_data['morphs']]
#
# train_data['morph_ids'] = morph_vocab.lookup_indices(train_data['morphs'])
# valid_data['word_ids'] = word_vocab.lookup_indices(valid_data['words'])
# valid_data['morph_ids'] = morph_vocab.lookup_indices(valid_data['morphs'])
# test_data['word_ids'] = word_vocab.lookup_indices(test_data['words'])
# test_data['morph_ids'] = morph_vocab.lookup_indices(test_data['morphs'])
#
# print("here")
# train_data = Dataset.from_dict(train_data)
# valid_data = Dataset.from_dict(valid_data)
# test_data = Dataset.from_dict(test_data)
#
# print("here2")
# # train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# # valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
# # test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)
#
# data_type = "torch"
# format_columns = ["word_ids", "morph_ids"]
#
# train_data = train_data.with_format(
#     type=data_type, columns=format_columns, output_all_columns=True
# )
#
# valid_data = valid_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )
#
# test_data = test_data.with_format(
#     type=data_type,
#     columns=format_columns,
#     output_all_columns=True,
# )

config = Config()
data = MorphemeDataLoader(config)

# def get_collate_fn(pad_index):
#     def collate_fn(batch):
#         batch_en_ids = [example["word_ids"] for example in batch]
#         batch_de_ids = [example["morph_ids"] for example in batch]
#         batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
#         batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
#         batch = {
#             "word_ids": batch_en_ids,
#             "morph_ids": batch_de_ids,
#         }
#         return batch
#
#     return collate_fn

# def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
#     collate_fn = get_collate_fn(pad_index)
#     data_loader = torch.utils.data.DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         collate_fn=collate_fn,
#         shuffle=shuffle,
#     )
#     return data_loader

batch_size = 128

# train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
# valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
# test_data_loader = get_data_loader(test_data, batch_size, pad_index)

class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hidden dim]
        # context = [n layers, batch size, hidden dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, embedding dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output = [seq length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # seq length and n directions will always be 1 in this decoder, therefore:
        # output = [1, batch size, hidden dim]
        # hidden = [n layers, batch size, hidden dim]
        # cell = [n layers, batch size, hidden dim]
        prediction = self.fc_out(output.squeeze(0))
        # prediction = [batch size, output dim]
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
                encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
                encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # first input to the decoder is the <sos> tokens
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hidden dim]
            # cell = [n layers, batch size, hidden dim]
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

input_dim = data.train.word_len
output_dim = data.train.morph_len
encoder_embedding_dim = 256
decoder_embedding_dim = 256
hidden_dim = 512
n_layers = 2
encoder_dropout = 0.5
decoder_dropout = 0.5
device = config.device()

encoder = Encoder(
    input_dim,
    encoder_embedding_dim,
    hidden_dim,
    n_layers,
    encoder_dropout,
)

decoder = Decoder(
    output_dim,
    decoder_embedding_dim,
    hidden_dim,
    n_layers,
    decoder_dropout,
)

model = Seq2Seq(encoder, decoder, device).to(device)
print(model)

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


model.apply(init_weights)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model):,} trainable parameters")

optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=data.train.pad_index)

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["morph_ids"].to(device)
        trg = batch["word_ids"].to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if i % 100 == 0:
            print(f"Epoch: {i}, loss: {loss}, epoch: {epoch_loss}")
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            src = batch["morph_ids"].to(device)
            trg = batch["word_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

n_epochs = 10
clip = 1.0
teacher_forcing_ratio = 0.5

best_valid_loss = float("inf")

for epoch in tqdm.tqdm(range(n_epochs)):
    train_loss = train_fn(
        model,
        data.train.loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device,
    )
    valid_loss = evaluate_fn(
        model,
        data.validation.loader,
        criterion,
        device,
    )
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), "tut1-model.pt")
    print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
    print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

    model.load_state_dict(torch.load("tut1-model.pt"))

    test_loss = evaluate_fn(model, data.test.loader, criterion, device)

    print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")


def segment_word(
        word,
        model,
        word_vocab,
        morph_vocab,
        sos_token,
        eos_token,
        device,
        max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        tokens = [sos_token] + word + [eos_token]
        ids = morph_vocab.lookup_indices(tokens)
        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        hidden, cell = model.encoder(tensor)
        inputs = word_vocab.lookup_indices([sos_token])
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == word_vocab[eos_token]:
                break
        tokens = word_vocab.lookup_tokens(inputs)
    return tokens

print("Checking first 10 trained examples")
print("=" * 80)
count = 20
for word, morph in zip(data.train.words[:count], data.train.morphs[:count]):
    pred = segment_word(word, model, data.train.word_vocab, data.train.morph_vocab, '<SOS>', '<EOS>', device)
    print(f'word: {"".join(word[1:-1])}\n\t  pred: {pred}\n\tactual: {"".join(morph[1:-1])}')

# expected_translation = test_data[0]["en"]
# sentence = test_data[0]["de"]
#
# print(sentence, expected_translation)
#
# translation = translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     de_nlp,
#     en_vocab,
#     de_vocab,
#     lower,
#     SOS_TOKEN,
#     EOS_TOKEN,
#     device,
# )
#
# sentence = "Ein Mann sitzt auf einer Bank."
#
# translation = translate_sentence(
#     sentence,
#     model,
#     en_nlp,
#     de_nlp,
#     en_vocab,
#     de_vocab,
#     lower,
#     SOS_TOKEN,
#     EOS_TOKEN,
#     device,
# )
#
# print(translation)
#
# translations = [
#     translate_sentence(
#         example["de"],
#         model,
#         en_nlp,
#         de_nlp,
#         en_vocab,
#         de_vocab,
#         lower,
#         SOS_TOKEN,
#         EOS_TOKEN,
#         device,
#     )
#     for example in tqdm.tqdm(test_data)
# ]
#
# bleu = evaluate.load("bleu")
#
# predictions = [" ".join(translation[1:-1]) for translation in translations]
#
# references = [[example["en"]] for example in test_data]
