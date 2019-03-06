# lang.py
# 2nd Mar. 2019
# IntSys-Summarization

C_PAD, C_PAD_IDX = "<PAD>", 0
C_UNK, C_UNK_IDX = "<UNK>", 1
C_SOS, C_SOS_IDX = "<SOS>", 2
C_EOS, C_EOS_IDX = "<EOS>", 3
MAX_SPECIAL_TOKEN_IDX = 3

# ======== CUSTOM EMBEDDINGS ========
def custom_word_indices(sentences):
    curr_idx = MAX_SPECIAL_TOKEN_IDX + 1
    vocab = {C_PAD : C_PAD_IDX, C_UNK : C_UNK_IDX, 
             C_SOS : C_SOS_IDX, C_EOS : C_EOS_IDX}

    idx_sentences = []

    for sent in sentences:
        temp_idx_sentence = []

        for word in sent:
            if word not in vocab:
                vocab[word] = curr_idx
                temp_idx_sentence.append(curr_idx)
                curr_idx += 1
            else:
                temp_idx_sentence.append(vocab[word])

        idx_sentences.append(temp_idx_sentence)

    return vocab, idx_sentences

# ======== PRETRAINED EMBEDDINGS ========
# def sentence_to_index(model, sentences):
#     idx_sentences = []

#     for sentence in sentences:
#         idx.append([model.wv.vocab.get(word).index if (word in model.wv.vocab) 
#                                                    else model.wv.vocab.get(UNK_TOKEN).index for word in sentence])

#     return idx_sentences

# # UNK : length, SOS : length + 1, EOS: length + 2, PAD : length + 3
# # Assuming weights is n x d
# # def add_special_vectors(weights):
# #     unk_vector = np.mean(weights, axis = 0)
# #     sos_vector = 

# #     pad_vectors = np.zeros(weights.shape[1]) 