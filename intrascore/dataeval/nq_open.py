import functools
import os
import sys

import datasets
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import settings as settings

@functools.lru_cache()
def get_fs_samples_prompt():
    data = datasets.load_dataset("nq_open", split='train')
    indices = np.random.RandomState(42).choice(len(data), 5) # 5개의 Q-A 랜덤 선택
    ret = ''
    for i in indices:
        i = int(i)
        ret += '\nQ: ' + data[i]['question'] + '\nA: ' + data[i]['answer'][0]
    return ret

def sample_to_prompt(sample, **kwargs): # five-shot prompting
    if isinstance(sample['question'], list): # sample['question']이 리스트라면
        return [sample_to_prompt({'question': _}, **kwargs) for _ in sample['question']] # 재귀적 반환
    return f"""Answer these questions:{get_fs_samples_prompt()} 
Q: {sample['question']}
A:"""

def generate_config(tokenizer):
    eos_token_id = [tokenizer(_)['input_ids'][-1] for _ in ['\n', ',', '.']]
    eos_token_id += [tokenizer.eos_token_id]
    bad_words_ids = [tokenizer(_)['input_ids'] for _ in ['Q:']] # only "Q"
    return dict(eos_token_id=eos_token_id, bad_words_ids=bad_words_ids)

def get_dataset(tokenizer):
    # For Natural Questions we use the test split used for open-domain question answering containing 3610 questions.
    data = datasets.load_dataset("nq_open", split='validation') # nq_open은 datasets 모듈에서 제공하는 데이터셋
    data.save_to_disk("/mnt/aix7101/minsuh-dataset/nq_open")
    id_map = {_['question']:str(i) for i, _ in enumerate(data)}

    def process_instance(example):
        example['id'] = id_map[example['question']]
        all_answers = example.pop('answer')
        print(f"answers: {all_answers}")
        print("\n\n")
        example['additional_answers'] = all_answers[1:]        
        example['answer'] = all_answers[0]

        example['prompt'] = sample_to_prompt({k:example[k] for k in ['question']})
        inputs = tokenizer(example['prompt'], padding=False, truncation=False)
        outputs = tokenizer(all_answers[0], padding=False, truncation=False)
        example['input_ids'] = inputs['input_ids']
        example["attention_mask"] = inputs.attention_mask
        example["labels"] = outputs.input_ids.copy()
        example["labels"] = [-100 if _ == tokenizer.pad_token_id else _ for _ in example["labels"]]
        return example
    data = data.map(process_instance, load_from_cache_file=False)
    data.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
        output_all_columns=True)
    return data

if __name__ == '__main__':
    import models

    tokenizer = models.load_pretrained_tokenizer('llama-7b-hf')
    data = get_dataset(tokenizer)