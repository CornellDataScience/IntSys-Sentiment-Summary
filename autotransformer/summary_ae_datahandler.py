import torch
from torchtext import data
from torch.autograd import Variable
from .transformer.functional import subsequent_mask
from .transformer.flow import batch_size_fn_inference
from .transformer.my_iterator import MyIterator


def whitespace_tokenizer(text):
    return text.strip().split()

def make_sentence_iterator(sent_list, batch_size):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    CLS_WORD = '<cls>'

    SRC = data.Field(tokenize=whitespace_tokenizer, lower=True, init_token=CLS_WORD, pad_token=BLANK_WORD)

    examples = []
    for s in sent_list:
        exp = data.Example.fromlist(s, [('src', SRC)])
        examples.append(exp)

    sent_data = data.Dataset(examples, [('src', SRC)])

    sent_iter = MyIterator(sent_data, batch_size=batch_size, device=0, repeat=False,
                            sort_key=lambda x: len(x.src), batch_size_fn=batch_size_fn_inference, train=False)

    return sent_iter, SRC, BOS_WORD, EOS_WORD, BLANK_WORD, CLS_WORD

def greedy_decode(model, encodings, trg_vocab, max_len=30, EOS_WORD='</s>'):
    #TODO: batch this
    candidates = []
    for candidate in encodings:
        candidate = torch.from_numpy(candidate).cuda()
        ys = torch.ones(1, 1).fill_(trg_vocab.stoi['<s>']).type(torch.LongTensor).cuda()
        for i in range(max_len - 1):
            out = model.decode(candidate.reshape(1,1,300), torch.ones(1,1,1).cuda(), Variable(ys),
                                 Variable(subsequent_mask(ys.size(1)).type(torch.LongTensor).cuda()))
            prob = model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data[0]
            ys = torch.cat([ys, torch.ones(1, 1).type(torch.LongTensor).fill_(next_word).cuda()], dim=1)
        sentence_words = []
        for k in range(1, ys.size(1)):
            sym = trg_vocab.itos[ys[0, k]]
            if sym == EOS_WORD:
                break
            sentence_words.append(sym)
        candidates.append(' '.join(sentence_words))
    return candidates

    