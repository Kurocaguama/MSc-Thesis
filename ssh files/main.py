import torch, huggingface_hub
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
from peft import LoraConfig, TaskType

hf_key = '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'
huggingface_hub.login(hf_key)

model_id = 'meta-llama/Llama-3.1-8B-Instruct' # Se modifica en función del modelo que se quiera implementar
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()


lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode= False,
    r = 8,
    lora_alpha=32,
    lora_dropout=0.1
)

quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)
generation_config = GenerationConfig.from_pretrained(model_id)

# Declaración de modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = """
    <|im_start|>system
    {SYSTEM}<|im_end|>
    <|im_start|>user
    {INPUT}<|im_ed|>
    <|im_start|>assistant
    {OUTPUT}<|im_end|>
"""

model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir = '/media/discoexterno/francisco/modelos', quantization_config = quant_config, generation_config = generation_config).to(dev)
model.add_adapter(lora_config, adapter_name = 'lora_godpls')
model.generation_config.pad_token_id = tokenizer.pad_token_id
tokenizer.pad_token = tokenizer.eos_token

# Dataset
beam_search = load_dataset('Kurosawama/beam_search_DPO', split = 'train')

# DPO args
training_args = DPOConfig(output_dir = 'media/discoexterno/francisco', logging_steps = 20)
trainer = DPOTrainer(model = model, args = training_args, processing_class = tokenizer, train_dataset = beam_search)
trainer.train()

#Los modelos resultantes están en huggingface para libre uso. 
model.push_to_hub('Llama-3.1-8B-Instruct-DPO-beamsearch-align') # El se modifica en función del DS y el modelo usado. 