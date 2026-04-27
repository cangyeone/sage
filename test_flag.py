from FlagEmbedding import BGEM3FlagModel
model = BGEM3FlagModel(
                "open_models/bge-m3",
                use_fp16=True,
                device="cpu",
            )
            