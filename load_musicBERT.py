import torch

class StandaloneMusicBERTModel(torch.nn.Module):
    def __init__(self, model, dictionary):
        super().__init__()
        self.model = model
        self.dictionary = dictionary

    def forward(self, grouped_input_tokens):
        """Convert grouped input tokens to tensor IDs and pass through the model."""
        token_ids = []
        for token_group in grouped_input_tokens:
            if len(token_group) != 8:
                raise ValueError("Each token group must contain exactly 8 tokens.")
            group_ids = [self.dictionary.index(token) for token in token_group]
            token_ids.extend(group_ids)
        input_tensor = torch.tensor(token_ids).unsqueeze(0)  # Add batch dimension
        return self.model(**{"src_tokens": input_tensor})

# class MusicBERT:
#     def __init__(self):
#         model_path = "standalone_musicbert_model.pth"
#         self.model = torch.load(model_path)

#     def test(self):
#         self.model.eval()

#         example_grouped_tokens = [
#             ["<s>", "<mask>", "<0-0>", "<1-0>", "<2-0>", "<3-0>", "<4-0>", "<5-0>"],
#             ["<6-0>", "<7-0>", "<mask>", "<0-1>", "<1-1>", "<2-1>", "<3-1>", "</s>"]
#         ]
    
#         output = self.model(example_grouped_tokens)
#         print("Output:", output , output[0].shape)

# if __name__ == "__main__":
#     model = MusicBERT()
#     model.test()

model_path = "standalone_musicbert_model.pth"
model = torch.load(model_path)

# def test():
#     model.eval()
#     example_grouped_tokens = [
#         ["<s>", "<mask>", "<0-0>", "<1-0>", "<2-0>", "<3-0>", "<4-0>", "<5-0>"],
#         ["<6-0>", "<7-0>", "<mask>", "<0-1>", "<1-1>", "<2-1>", "<3-1>", "</s>"]
#     ]
#     output = model(example_grouped_tokens)
#     print("Output:", output , output[0].shape)
