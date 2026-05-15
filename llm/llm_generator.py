import torch

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM


class LLMGenerator:

    def __init__(self):

        model_name = "Qwen/Qwen2-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32
        )

        self.model.eval()

    def generate_reasoning(self, state_text):

        prompt = f"""
        You are controlling an agent in MiniGrid.

        Current environment:
        {state_text}

        Generate ONLY one short action strategy.

        Correct examples:
        Find the key first.
        Open the door carefully.
        Move toward the goal.
        Explore another room.

        Wrong examples:
        The agent should think carefully.
        Choose a strategy.
        Here are some possible actions.
        Notes: the agent may need a key.

        Now generate one short strategy:

        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        with torch.no_grad():

            outputs = self.model.generate(

                **inputs,

                max_new_tokens=10,

                do_sample=True,

                top_p=0.9,

                temperature=0.6
            )

        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        # prompt 제거
        reasoning = generated_text.replace(
            prompt,
            ""
        ).strip()

        # 첫 줄만 사용
        reasoning = reasoning.split("\n")[0]

        # bullet 제거
        reasoning = reasoning.replace("-", "").strip()

        # 너무 길면 자르기
        reasoning = reasoning[:60]

        # 번호 제거
        reasoning = reasoning.replace("1.", "").strip()

        return reasoning