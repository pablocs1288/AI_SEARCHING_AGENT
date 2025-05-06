
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline



class Embeddings:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state.mean(dim=1)
        return last_hidden.cpu().numpy().tolist() # torch tensor -> numpy

    def embed_query(self, text):
        return self.embed([text])[0]

    def embed_documents(self, texts):
        return self.embed(texts)
    