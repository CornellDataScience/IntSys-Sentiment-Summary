"""An example config dictionary for the whole summarization pipeline"""

config = {
    'dataset_path':None,
    'dataset':None,

    'extractive': False,
    'device':None,

    'src_vocab_path': 'autotransformer/models/electronics/src_vocab.pt',
    'src_vocab' : None,
    'trg_vocab_path' : 'autotransformer/models/electronics/tgt_vocab.pt',
    'autoencoder_path': 'autotransformer/models/electronics/electronics_autoencoder_epoch7_weights.pt',
    'autoencoder':None,
    'ae_batchsize': 5000,

    'density_parameter': 2,
    'minimum_samples': 2,
    'min_clusters': 50,
    'max_acceptable_clusters':100,
    'min_num_candidates': 200,

    'BERT_finetune_path':'bert_finetune/models/finetune_electronics_mae1.pt',
    'BERT_config_path': 'bert_finetune/models/finetune_electronics_mae1config.json',
    'BERT_finetune_model': None,
    'BERT_batchsize': 100,

    'opt_function' : None,
    'opt_dict' : {
        'sentence_cap': 20,
        'n_elite':5,
        'init_pop': 96,

    } 
}