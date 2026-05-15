from llm.llm_generator import LLMGenerator


generator = LLMGenerator()

state_text = "A key is visible."

reasoning = generator.generate_reasoning(
    state_text
)

print("\nGenerated Reasoning:")
print(reasoning)