import torch, trl, huggingface_hub
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

hf_key = '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'
huggingface_hub.login(hf_key)

model_id = 'meta-llama/Llama-3.2-1B-Instruct'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generation_config = GenerationConfig.from_pretrained(model_id)

# Declaración de modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = '/media/discoexterno/francisco/modelos', generation_config = generation_config).to(dev)
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.pad_token = tokenizer.eos_token

# Dataset
beam_search = load_dataset('Kurosawama/beam_search_DPO', split = 'train')

# DPO args
training_args = DPOConfig(output_dir = 'media/discoexterno/francisco', logging_steps = 20)
trainer = DPOTrainer(model = model, args = training_args, processing_class = tokenizer, train_dataset = beam_search)
trainer.train()
model.push_to_hub('Llama-3.2-1B-Instruct-DPO-beamsearch-align')