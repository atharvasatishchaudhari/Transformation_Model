import math
import torch
import torch.nn as nn
import gradio as gr
from model import build_transformer
from config_file import get_config, get_weights_file_path
from tokenizers import Tokenizer

# Define the casual mask function (as used in training)
def casual_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0

# Greedy decode function for inference
def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')
    
    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0).unsqueeze(0)], dim=1)
        if next_word.item() == eos_idx:
            break
    return decoder_input.squeeze(0)

# Function to load a tokenizer from file
def load_tokenizer(tokenizer_file):
    return Tokenizer.from_file(tokenizer_file)

# Main translation function that will be called by Gradio
def translate_text(input_text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = get_config()
    # Load tokenizers (assumes they have been built and saved during training)
    tokenizer_src_file = config["tokenizer_file"].format(config["lang_src"])
    tokenizer_tgt_file = config["tokenizer_file"].format(config["lang_tgt"])
    
    try:
        tokenizer_src = load_tokenizer(tokenizer_src_file)
        tokenizer_tgt = load_tokenizer(tokenizer_tgt_file)
    except Exception as e:
        return f"Error loading tokenizers: {e}"
    
    # Build model using vocab sizes from tokenizers
    src_vocab_size = tokenizer_src.get_vocab_size()
    tgt_vocab_size = tokenizer_tgt.get_vocab_size()
    model = build_transformer(src_vocab_size, tgt_vocab_size, config["seq_len"], config["seq_len"], d_model=config["d_model"])
    model.to(device)
    
    # Load trained weights (assumes final model is saved as last epoch weight)
    weights_file = get_weights_file_path(config, f"{config['num_epochs']-1:02d}")
    try:
        state = torch.load(weights_file, map_location=device)
    except Exception as e:
        return f"Error loading model weights: {e}"
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    
    # Tokenize input text (for the source language)
    sos_id = tokenizer_src.token_to_id("[SOS]")
    eos_id = tokenizer_src.token_to_id("[EOS]")
    encoded = tokenizer_src.encode(input_text).ids
    seq_len = config["seq_len"]
    num_padding_tokens = seq_len - len(encoded) - 2
    if num_padding_tokens < 0:
        return "Error: Input sentence too long."
    encoder_input_ids = [sos_id] + encoded + [eos_id] + [tokenizer_src.token_to_id("[PAD]")] * num_padding_tokens
    encoder_input = torch.tensor(encoder_input_ids, dtype=torch.int64).unsqueeze(0).to(device)
    encoder_mask = (encoder_input != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)
    
    # Generate translation using greedy decoding
    decoder_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, seq_len, device)
    # Remove the initial SOS token from the output
    output_ids = decoder_output.tolist()[1:]
    translation = tokenizer_tgt.decode(output_ids)
    return translation

# Create a Gradio interface
iface = gr.Interface(
    fn=translate_text,
    inputs=gr.inputs.Textbox(lines=2, placeholder="Enter an English sentence here"),
    outputs="text",
    title="Transformer Translation (EN to IT)",
    description="Enter an English sentence to translate it to Italian using the trained transformer model."
)

if __name__ == "__main__":
    iface.launch()
