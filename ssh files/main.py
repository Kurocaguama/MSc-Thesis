import torch, trl, huggingface_hub
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

model_id = 'meta-llama/Llama-3.2-1B-Instruct'

generation_config = GenerationConfig.from_pretrained(model_id)

# Declaración de modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map = '/media/discoexterno/francisco', generation_config = generation_config)
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.pad_token = tokenizer.eos_token

# Dataset
beam_search = load_dataset('Kurosawama/beam_search_DPO', split = 'train')

# DPO args
training_args = DPOConfig(output_dir = 'media/discoexterno/francisco', logging_steps = 20)
trainer = DPOTrainer(model = model, args = training_args, processing_class = tokenizer, train_dataset = beam_search)
trainer.train()
model.push_to_hub('Llama-3.2-1B-Instruct-DPO-beamsearch-align')