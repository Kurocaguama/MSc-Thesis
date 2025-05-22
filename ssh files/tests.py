import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GenerationConfig
from datasets import load_dataset

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
kuro_ds = load_dataset('Kurosawama/beam_search_DPO', split = 'train')
folio = load_dataset('yale-nlp/FOLIO', split='validation')
print(dev)

def initialize_model(model_id, llama3):
    quant_config = BitsAndBytesConfig(load_in_4bit = True, bnb_4bit_compute_dtype = torch.bfloat16)
    gen_config = GenerationConfig.from_pretrained(model_id)
    if llama3:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B')
    else:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    tokenizer.chat_template = """
            <|im_start|>system
            {SYSTEM}<|im_end|>
            <|im_start|>user
            {INPUT}<|im_ed|>
            <|im_start|>assistant
            {OUTPUT}<|im_end|>
        """
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config = quant_config, generation_config = gen_config).to(dev)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer

base_model_id = 'meta-llama/Llama-2-7b-hf'
align_model_id = 'Kurosawama/Llama-2-7b-DPO-beamsearch-align'

base_model, base_tokenizer = initialize_model(base_model_id, False)
align_model, align_tokenizer = initialize_model(align_model_id, False)

def answer(prompt, model, tokenizer):
    """
    Genera la respuesta del modelo siguiendo la estrategia de beam_search.
    """
    inputs = tokenizer(prompt, return_tensors = 'pt').to(dev)
    outputs = model.generate(**inputs, max_new_tokens = 200, num_beams = 3)
    ans = tokenizer.batch_decode(outputs, skip_special_tokens = True)[0]
    ans = ans[len(prompt):]
    return ans


prompt_ex = """
Given a problem description and a question. The task is to parse the problem and the question into first-order logic formulars.
    The grammar of the first-order logic formular is defined as follows:
    1) logical conjunction of expr1 and expr2: expr1 ∧ expr2
    2) logical disjunction of expr1 and expr2: expr1 ∨ expr2
    3) logical exclusive disjunction of expr1 and expr2: expr1 ⊕ expr2
    4) logical negation of expr1: ¬expr1
    5) expr1 implies expr2: expr1 → expr2
    6) expr1 if and only if expr2: expr1 ↔ expr2
    7) logical universal quantification: ∀x
    8) logical existential quantification: ∃x
    --------------
    Problem:
    All people who regularly drink coffee are dependent on caffeine. People either regularly drink coffee or joke about being addicted to caffeine. No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    Predicates:
    Dependent(x) ::: x is a person dependent on caffeine.
    Drinks(x) ::: x regularly drinks coffee.
    Jokes(x) ::: x jokes about being addicted to caffeine.
    Unaware(x) ::: x is unaware that caffeine is a drug.
    Student(x) ::: x is a student.
    Premises:
    ∀x (Drinks(x) → Dependent(x)) ::: All people who regularly drink coffee are dependent on caffeine.
    ∀x (Drinks(x) ⊕ Jokes(x)) ::: People either regularly drink coffee or joke about being addicted to caffeine.
    ∀x (Jokes(x) → ¬Unaware(x)) ::: No one who jokes about being addicted to caffeine is unaware that caffeine is a drug. 
    (Student(rina) ∧ Unaware(rina)) ⊕ ¬(Student(rina) ∨ Unaware(rina)) ::: Rina is either a student and unaware that caffeine is a drug, or neither a student nor unaware that caffeine is a drug. 
    ¬(Dependent(rina) ∧ Student(rina)) → (Dependent(rina) ∧ Student(rina)) ⊕ ¬(Dependent(rina) ∨ Student(rina)) ::: If Rina is not a person dependent on caffeine and a student, then Rina is either a person dependent on caffeine and a student, or neither a person dependent on caffeine nor a student.
    --------------
    
    Problem:
    {}
    Predicates:
"""


print(answer(prompt_ex.format(folio['premises'][1]), base_model, base_tokenizer))
print("---------------------")
print(answer(prompt_ex.format(folio['premises'][1]), align_model, align_tokenizer))