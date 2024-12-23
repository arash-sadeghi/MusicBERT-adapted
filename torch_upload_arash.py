import torch
# from transformers import RobertaModel, RobertaConfig
from musicbert.__init__ import MusicBERTModel
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
print(roberta)
exit(0)
# Define the path to your weights file
weight_path = "checkpoint_last_musicbert_base.pt"  # Replace with the actual path

# Load a RoBERTa configuration
config = RobertaConfig.from_pretrained("roberta-base")  # Use 'roberta-large' or another variant if needed

# Initialize the RoBERTa model
model = RobertaModel(config)

print (model.encoder.layer[0].attention.self.query.weight)
#! str(roberta.state_dict().keys())[0:200]
# # Load weights from the provided path
# try:
#     state_dict = torch.load(weight_path, map_location=torch.device("cpu"))  # Adjust device if using GPU
#     model.load_state_dict(state_dict, strict=False)  # `strict=False` allows partial loading
#     print("Weights successfully loaded into the model.")
# except Exception as e:
#     print(f"Error loading weights: {e}")
#! str(state_dict['model'].keys())[0:200]
state_dict = torch.load(weight_path, map_location=torch.device("cpu"))  # Adjust device if using GPU
model.load_state_dict(state_dict, strict=True)  # `strict=False` allows partial loading
print("Weights successfully loaded into the model.")
print(model.encoder.layer[0].attention.self.query.weight)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Test the model with dummy input
# from transformers import RobertaTokenizer

# tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# dummy_input = tokenizer("This is a test sentence.", return_tensors="pt").to(device)
# output = model(**dummy_input)

# print("Model output shape:", output.last_hidden_state.shape)
