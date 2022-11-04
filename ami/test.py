from nemo.collections.asr.modules.ex_scconformer_encoder import SelfConditionedExemplarConformerEncoder

if __name__ == "main":
    encoder = SelfConditionedExemplarConformerEncoder(
        feat_in = 1, 
        n_layers = 1,
        d_model = 1
    )