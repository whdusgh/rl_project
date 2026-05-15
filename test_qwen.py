from llm.qwen_encoder import QwenEncoder

encoder = QwenEncoder()

text = "Find the key first."

embedding = encoder.encode(text)

print("Embedding shape:")
print(embedding.shape)

print("\nEmbedding:")
print(embedding)