import operator

from datasets import load_from_disk
import json

# ds = load_from_disk("D:/University/trsshi/filtered_dataset")
# size_before_amount = 0
# size_before_bytes = 0

# for data in ds:
#     size_before_amount += 1
#     size_before_bytes += len(data["text"].encode("utf-8"))

# print("Amount of documents before cleaning and filtering: " + str(size_before_amount))
# print("Total size in bytes before cleaning and filtering: " + str(size_before_bytes))


# size_after_amount = 0
# size_after_bytes = 0

# with open('articles_checked.jsonl', 'r', encoding='utf-8') as f:
#     for line in f:
#         line = line.strip()
#         if line: 
#             data = json.loads(line)
#             size_after_amount += 1
#             size_after_bytes += len(data["text"].encode("utf-8"))

# print("Amount of documents after cleaning and filtering: " + str(size_after_amount))
# print("Total size in bytes after cleaning and filtering: " + str(size_after_bytes))


amount_of_chunks = 0
total_size_of_chunks_symbol = 0
total_size_of_chunks_words = 0

words_in_chunks = dict()
words_in_chunks_no_punctuation = dict()
words_in_chunks_lower = dict()
words_in_chunks_lower_no_punctuation = dict()

with open('data_chunks.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line: 
            text = json.loads(line)['chunk_text']
            amount_of_chunks += 1
            total_size_of_chunks_symbol += len(text)

            words = text.split()
            total_size_of_chunks_words += len(words)
            for word in words:
                words_in_chunks[word] = words_in_chunks.get(word, 0) + 1

            punctuation = str.maketrans("", "", "#$%&()*+,./:;<=>?@[]^_`{|}~")
            words_no_punctuation = list(map(lambda w: str.translate(w, punctuation), words))
            for word in words_no_punctuation:
                words_in_chunks_no_punctuation[word] = words_in_chunks_no_punctuation.get(word, 0) + 1

            words_lower = list(map(lambda w: str.lower(w), words))
            for word in words_lower:
                words_in_chunks_lower[word] = words_in_chunks_lower.get(word, 0) + 1

            words_no_punctuation_lower = list(map(lambda w: str.translate(w, punctuation), words_lower))
            for word in words_no_punctuation_lower:
                words_in_chunks_lower_no_punctuation[word] = words_in_chunks_lower_no_punctuation.get(word, 0) + 1

print("Amount of chunks: " + str(amount_of_chunks))
print("Mean size in symbols: " + str(total_size_of_chunks_symbol / amount_of_chunks))
print("Mean size in words: " + str(total_size_of_chunks_words / amount_of_chunks))

sorted_words_in_chunks = sorted(words_in_chunks.items(), key=operator.itemgetter(1), reverse=True)
print("Top 20 words in chunks: " + str(sorted_words_in_chunks[:20]))

sorted_words_in_chunks_no_punctuation = sorted(words_in_chunks_no_punctuation.items(), key=operator.itemgetter(1), reverse=True)
print("Top 20 words in chunks without punctuation: " + str(sorted_words_in_chunks_no_punctuation[:20]))

sorted_words_in_chunks_lower = sorted(words_in_chunks_lower.items(), key=operator.itemgetter(1), reverse=True)
print("Top 20 words in chunks after lower: " + str(sorted_words_in_chunks_lower[:20]))

sorted_words_in_chunks_lower_no_punctuation = sorted(words_in_chunks_lower_no_punctuation.items(), key=operator.itemgetter(1), reverse=True)
print("Top 20 words in chunks after lower without punctuation: " + str(sorted_words_in_chunks_lower_no_punctuation[:20]))