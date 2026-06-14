from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

device = "cuda:1"

qwen_model_id = "Qwen/Qwen2.5-3B-Instruct"##"Qwen/Qwen3-4B"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=torch.float16).to(device)


languages = ["french", "english", "spanish", "italien", "german", "portuguese", "greek", "russian", "japanese", "hindi", "chinese", "dutch", "egyptian arabic", "thai", "lingala", "senegalese"]

def translate_word(text):
    prompt = f"""
    Translate the following word '{text}' in the following languages : {languages}

    One line in the format :
    french | english | spanish | italien | german | portuguese | greek | russian | japanese | hindi | chinese | dutch | egyptian arabic | thai | lingala | senegalese

    No explanations.
    """

    messages = [
    {"role": "user", "content": prompt}
    ]
    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = qwen_tokenizer([text], return_tensors="pt").to(device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=512 ##4096 pour thinking
    )
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = qwen_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


words = ["chat", "météo", "chaussette", "ergonomie", "orteil", "liseuse", "modification", "grand", "voile", "rayure", "dessert",  "désert", "chocolat", "clavier", "noir", "bateau", "arbre", "chaine", "lys", "virus	", "créole", "bague", "betterave", "électricité", "tambour"]

translations = []

for w in words :
    t= translate_word(w)
    print(t)
    #translations.append(t.split(' | '))

#print(translations)

#df = pd.DataFrame(translations, columns=languages)

#df.to_csv('data/words.csv')

