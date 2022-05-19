hidden_dim = 100
hp_config = {
    "vocab_size": 58036,
    "num_labels": 8929,
    "word_embeddings_dim": 100,
    "hidden_dim": hidden_dim,
    "word_embeddings_path": './preprocessed_mimic3/word_embeddings.npy',

    "attention_dropout_rate": 0.2,
    "semantic_dropout_rate": 0.2,
    "num_attention_heads": 1,

    "rnn_config": {
        "hidden_dim": hidden_dim,
        "num_layers": 1,
        "output_dim": hidden_dim,
        "dropout_rate": 0.2,
    },
}