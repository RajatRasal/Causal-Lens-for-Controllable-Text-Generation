from torchtext.datasets import YelpReviewPolarity

# TODO: Re-train the model, with same learning rates, on new dataset
# but with beta_cycle_len = 1.

dataset = YelpReviewPolarity(root="./data", split="train")
for row in dataset:
    print(row)
    break
