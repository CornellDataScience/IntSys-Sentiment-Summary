"""The config dictionary for the whole summarization pipeline"""

config = {
  'abstractive': True,

  'scr_vocab_path': None,
  'scr_vocab' : None,
  'trg_vocab_path' : None,
  'model_path': None,
  'encode_model': None,
  'decode_model': None,

  'density_parameter': .04,
  'minimum_samples': 4,
  'min_clusters': 5,
  'max_acceptable_clusters':30,
  'min_num_candidates': 100,

  'opt_function' : None,
  'opt_dict' = {
    'sentence_cap': 20,
    'n_elite':5,
    'init_pop': 96,

  }
}