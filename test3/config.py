class Config:
    model = "SVM"
    runs = 1  # No. of runs of experiments

    # Training modes
    use_context = False  # whether to use context information or not (default false)
    use_author = False  # add author one-hot encoding in the input

    use_bert = True  # if False, uses glove pooling

    use_target_text = True
    use_target_audio = True  # adds audio target utterance features.
    use_target_video = True  # adds video target utterance features.

    speaker_independent = False  # speaker independent experiments

    embedding_dim = 300  # GloVe embedding size
    word_embedding_path = "/home/sacastro/glove.840B.300d.txt"
    max_sent_length = 20
    max_context_length = 4  # Maximum sentences to take in context
    num_classes = 2  # Binary classification of sarcasm

    svm_c = 10.0
    svm_scale = True

    fold = None


class SpeakerDependentTConfig(Config):
    use_target_text = True
    svm_c = 1.0


class SpeakerDependentAConfig(Config):
    use_target_audio = True
    svm_c = 1.0


class SpeakerDependentVConfig(Config):
    use_target_video = True
    svm_c = 1.0

CONFIG_BY_KEY = {
    "": Config(),
    "t": SpeakerDependentTConfig(),
    "a": SpeakerDependentAConfig(),
    "v": SpeakerDependentVConfig(),
}
