import transformers
from deepeval.models.base_model import DeepEvalBaseLLM
from typing import List
import torch
from vllm import SamplingParams

class LLMLoad(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer,
        device='cuda',  # Default to 'cuda'; adjust as needed
        use_bfloat16=True  # Flag to enable bfloat16 precision
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_bfloat16 = use_bfloat16

    def load_model(self):
        model = self.model
        return model
        # if self.use_bfloat16:
        #     model = model.to(dtype=torch.bfloat16)  # Convert model to bfloat16 precision
        # return model.to(self.device)

    def generate(self, prompt: str) -> str: #edited by aojun
        model = self.load_model()
        sampling_params = SamplingParams(temperature=0.2, top_k=5, max_tokens=1000)
        outputs = model.generate([prompt], sampling_params=sampling_params)
        response = ''
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        response = generated_text
        return response


        

    def generate_one_completion(self,prompt: str) -> str:
        """
        Generate a single completion for the given prompt using the loaded language model.
        
        Args:
            prompt (str): The input prompt to generate a completion for.
            
        Returns:
            str: The generated completion.
        """
        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
        
        # Generate the completion
        outputs =  model.generate([prompt], sampling_params=sampling_params)
        
        # Decode the output into text
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt to return only the generated completion
        return completion[len(prompt):].strip()
        
    def generate_samples(
        self, prompt: str, n: int, temperature: float
    ):
        completions = []
        model = self.load_model()
        sampling_params = SamplingParams(temperature=0.2, top_k=5, max_tokens=1000)
        outputs = model.generate([prompt], sampling_params=sampling_params)
        response = ''
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
        response = generated_text
        completions.append(response)
        return completions



    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        model = self.load_model()

        sampling_params = SamplingParams(temperature=0.2, top_k=5, max_tokens=1000)
        outputs = model.generate(prompts, sampling_params=sampling_params)

        return outputs

        # model_inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        # model.to(self.device)

        # generated_ids = model.generate(**model_inputs, max_new_tokens=100, do_sample=True)
        # return self.tokenizer.batch_decode(generated_ids)

    def get_model_name(self):
        return "LLM Model"
