from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"

qwen_model_id = "Qwen/Qwen2.5-7B-Instruct"##"Qwen/Qwen3-4B"
qwen_tokenizer = AutoTokenizer.from_pretrained(qwen_model_id)
qwen_model = AutoModelForCausalLM.from_pretrained(
    qwen_model_id,
    dtype=torch.float16).to(device)


list_relations = {
'FormOf' : "'subject' is an inflected form of 'object'; 'object' is the root word of 'subject'" ,
'IsA' : "'subject' is a subtype or a specific instance of 'object'" ,
'PartOf' : "'subject' is a part of 'object'" ,
'HasA' : "'object' belongs to 'subject'" ,
'Contains': "'subject' contains 'object'",
'UsedFor' : "'subject' is used for 'object'; the purpose of 'subject' is 'object'" ,
'CapableOf' : "Something that 'subject' can typically do is 'object'" ,
'AtLocation' : "'subject' is located at 'object', 'subject' can be an event taking place at 'object'" ,
'Causes' : "'subject' and 'object' are events, and 'subject' causes 'object'" ,
'HasSubevent' : "'subject' and 'object' are events, and 'object' happens as a subevent of 'subject'" ,
'HasPrerequisite' : "In order for 'subject' to happen, 'object' needs to happen; 'object' is a dependency of 'subject'" ,
'HasProperty' : "'subject' has 'object' as a property; 'subject' can be described as 'object'" ,
'MotivatedByGoal' : "Someone does 'subject' because they want result 'object'; 'subject' is a step toward accomplishing the goal 'object'" ,
'CreatedBy' : "'object' is a process or agent that creates 'subject'" ,
'Synonym' : "'subject' and 'object' have very similar meanings. Symmetric" ,
'Antonym' : "'subject' and 'object' are opposites in some relevant way" ,
'SymbolOf' : "'subject' symbolically represents 'object'" ,
'SimilarTo' : "'subject' is similar to 'object'. Symmetric" ,
'MadeOf' : "'subject' is made of 'object'" ,
'ReceivesAction' : "'object' can be done to 'subject'",
'PerformsAction' : "'subject' is doing the action 'object', 'object' is usually a verb."}





def extract_triplets_qwen(text):
    prompt = f"""
    Extract all factual (subject, relation, object) triples from sentence. 
    One triple per line in the format: 
    subject | relation | object

    No explanations. If no triple can be extracted, write nothing. 

    Here are the only relations allowed : {list_relations}
    You must ONLY use the relations provided in the allowed list. Do not invent any other relation.

    Here are examples of the expected decomposition behavior:

    Sentence: John is eating a ham sandwich
    John | PerformsAction | eating
    ham sandwich | ReceivesAction | eating

    Sentence: Frank is in New York, but he plays football.
    Frank | AtLocation | New York
    He | PerformsAction | plays
    football | ReceivesAction | plays

    Now extract the triples for this sentence:
    Sentence: {text}"""

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


phrases = ["He doesn't play baseball.", "Chris doesn't like staying with us.","John is eating a ham sandwich", "Two boys are playing outside", "A man is playing an instrument.", "Steve Jobs founded Apple.", "He and his friend played at the park, they played football and basketball.","Frank is in New York, but he plays football in Manchester.","The cat and the dog are in the kitchen.", "A cat and a dog are in a kitchen.", "Frank's dog doesn't eat fruits, he is allergic","Simon didn't call me back, he is busy.", "Younes is working on a project. His friend is playing a video game.", "I have never seen anyone like Frank, he must be gifted.","He must be sick.", "He is sick.", "He played football after eating.","A cat and a dog are in the kitchen","Two dogs are eating and playing football"]


for p in phrases :
    print("-"*50)
    print(f"{p} :")
    print(extract_triplets_qwen(p))