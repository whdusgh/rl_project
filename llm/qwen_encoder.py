import torch

from transformers import AutoTokenizer
from transformers import AutoModel


class QwenEncoder:

    def __init__(self):

        model_name = "Qwen/Qwen2-0.5B"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        self.model.eval()

    def encode(self, text):

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        with torch.no_grad():

            outputs = self.model(**inputs)

        # 마지막 hidden state 평균 pooling
        hidden_states = outputs.last_hidden_state

        embedding = hidden_states.mean(dim=1)

        return embedding.squeeze(0)