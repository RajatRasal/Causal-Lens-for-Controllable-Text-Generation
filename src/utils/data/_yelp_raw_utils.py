"""
Including categories of a business within the Yelp review document.

Ground truth json files downloaded from Kaggle Yelp Dataset:
https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset?resource=download&select=yelp_academic_dataset_review.json
"""
import json

from tqdm import tqdm

biz_to_cat = {}
with open("/Users/work/Downloads/yelp_academic_dataset_business.json") as f:
    for line in tqdm(f):
        try:
            biz = json.loads(line)
        except Exception:
            pass
        else:
            biz_to_cat[biz["business_id"]] = biz["categories"]

results = []
with open("/Users/work/Downloads/yelp_academic_dataset_review.json") as f:
    for line in tqdm(f):
        try:
            review = json.loads(line)
        except Exception:
            pass
        else:
            results.append(
                {
                    "text": review["text"],
                    "categories": biz_to_cat[review["business_id"]],
                }
            )

with open("./reviews_with_categories.json", "w") as f:
    json.dump(results, f)
