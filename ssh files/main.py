import torch, trl, huggingface_hub
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig

hf_key = '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'
huggingface_hub.login(hf_key)

model_id = 'meta-llama/Llama-2-7b-hf'
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)
generation_config = GenerationConfig.from_pretrained(model_id)

# Declaración de modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = '/media/discoexterno/francisco/modelos', quantization_config = quant_config, generation_config = generation_config).to(dev)
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.pad_token = tokenizer.eos_token

# Dataset
beam_search = load_dataset('Kurosawama/beam_search_DPO', split = 'train')

# DPO args
training_args = DPOConfig(output_dir = 'media/discoexterno/francisco', logging_steps = 20)
trainer = DPOTrainer(model = model, args = training_args, processing_class = tokenizer, train_dataset = beam_search)
trainer.train()
model.push_to_hub('Llama-2-7b-DPO-beamsearch-align')