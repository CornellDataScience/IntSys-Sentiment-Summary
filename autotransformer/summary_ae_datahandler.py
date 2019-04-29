from torchtext import data
from transformer.flow import batch_size_fn_inference
from transformer.my_iterator import MyIterator



def whitespace_tokenizer(text):
    return text.strip().split()

def make_sentence_iterator(sent_list, device):
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = '<blank>'
    CLS_WORD = '<cls>'

    SRC = torchtext.data.Field(tokenize=whitespace_tokenizer, lower=True, init_token=CLS_WORD, pad_token=BLANK_WORD)

    examples = []
    for s in sent_list:
        exp = torchtext.data.Example.fromlist(s, [('src', SRC)])
        examples.append(exp)

    sent_data = torchtext.data.Dataset(examples, [('src', SRC)])

    sent_iter = MyIterator(sent_data, batch_size=BATCH_SIZE, device=device, repeat=False,
                            sort_key=lambda x: len(x.src), batch_size_fn=batch_size_fn_inference, train=False)

    return sent_iter, SRC, BOS_WORD, EOS_WORD, BLANK_WORD, CLS_WORD

    