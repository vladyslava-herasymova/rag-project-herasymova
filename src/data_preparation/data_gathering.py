from datasets import load_dataset

ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", cache_dir="D:/University/trsshi/datasets")
print('Loaded dataset')
filtered = ds.filter(lambda example: ((str.lower(example["title"]).endswith("grand prix") 
                     or str.lower(example["title"]).endswith("formula one world championship")
                     or str.lower(example["title"]).endswith("formula one season"))
                     and ("formula one" in str.lower(example["text"])
                         or "formula 1" in str.lower(example["text"]))))
filtered.save_to_disk("D:/University/trsshi/filtered_dataset")
print(filtered)