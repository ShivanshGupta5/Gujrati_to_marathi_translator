# from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
# import torch

# # Load from disk
# model = M2M100ForConditionalGeneration.from_pretrained("./gu_mr_translator")
# tokenizer = M2M100Tokenizer.from_pretrained("./gu_mr_translator")

# # Move model to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# def translate(text):
#     tokenizer.src_lang = "guj_Gujr"  # Gujarati input
#     inputs = tokenizer(text, return_tensors="pt").to(device)

#     forced_bos_token_id = tokenizer.convert_tokens_to_ids("<mar_Deva>")  # Marathi output

#     output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# translate("હેલો કેમ છો")

from transformers import M2M100ForConditionalGeneration, NllbTokenizer
import torch

# Load from disk (correct tokenizer class)
tokenizer = NllbTokenizer.from_pretrained("./gu_mr_translator")
model = M2M100ForConditionalGeneration.from_pretrained("./gu_mr_translator")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def translate(text):
    tokenizer.src_lang = "guj_Gujr"  # Gujarati input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids("<mar_Deva>")  # Marathi output

    output = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test
print(translate("હેલો કેમ છો"))
