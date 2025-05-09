from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load model and tokenizer once
model_path = './checkpoint-2250'

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_recipe(ingredients_list, max_length=300):
    """
    Generate a recipe given a list of ingredients.
    Returns the generated text as a string.
    """
    prompt = f"Ingredients: {', '.join(ingredients_list)}\nRecipe Name:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    output = model.generate(
        inputs.input_ids,
        # max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.95,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    generated_text = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    return generated_text



# print(generate_recipe(["tomato", "broccoli", "grape"]))