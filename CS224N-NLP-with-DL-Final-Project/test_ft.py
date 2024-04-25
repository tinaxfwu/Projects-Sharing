import torch 
import torch.nn as nn
import numpy as np
# import torch_dct as dct
from scipy.fft import dct, idct


# # Adapted from https://github.com/google-research/google-research/blob/master/f_net/fourier.py
# def fftn(x):
#     """
#     Applies n-dimensional Fast Fourier Transform (FFT) to input array.

#     Args:
#         x: Input n-dimensional array.

#     Returns:
#         n-dimensional Fourier transform of input n-dimensional array.
#     """
#     out = x
#     # for axis in reversed(range(x.ndim)[1:]):  # We don't need to apply FFT to last axis
#     out = torch.fft.fft(out)
#     return out


# a = np.array([1,2,3,4,5])
# b = np.array([17, 17, 83, 101, 224])
# c = np.array([1,1,1,1,1])

# a = torch.tensor(a, dtype=torch.float32)
# b = torch.tensor(b, dtype=torch.float32)
# c = torch.tensor(c, dtype=torch.float32)

# aft = fftn(a)
# bft = fftn(b)
# cft = fftn(c)
# print(aft)
# print(bft)
# print(cft)

# # print(aft.real)
# # print(bft.real)
# # print(cft.real)

# cos = nn.CosineSimilarity(dim=0)
# print(cos(aft.real, bft.real))
# print(cos(a, b))
# print(cos(aft.real, cft.real))
# print(cos(a, c ))
# print(cos(bft.real, cft.real))
# print(cos(b, c ))



# # Try using FFT for embedding 
import spacy
# import numpy as np

# # Load the spaCy model for POS tagging
nlp = spacy.load("en_core_web_sm")

# # Example sentences
sentences = [
    "I am very very excited about the upcoming trip.",
    "I am super super excited about the upcoming trip.",
    # "I am sad today.",
    # "I am very excited about the upcoming trip.",
    # "I am happy yesterday.",
    # "I am happy yesterday.",

]

# Function to get POS tags
def get_pos_tags(sentence):
    doc = nlp(sentence)
    return [token.pos_ for token in doc]

# Encode POS tags numerically (simple example, real implementation might need a more robust approach)
pos_to_num = {
    "ADJ": 1, "ADV": 2, "INTJ": 3, "NOUN": 4, "PROPN": 5,
    "VERB": 6, "ADP": 7, "AUX": 8, "CONJ": 9, "DET": 10,
    "NUM": 11, "PART": 12, "PRON": 13, "SCONJ": 14,
    "PUNCT": 15, "SYM": 16, "X": 17, "SPACE": 18
}

# # Apply FFT to POS tag sequences
# def apply_fft_to_pos(sentence, max_length):
#     print(f"Sentence: {sentence}")
#     pos_tags = get_pos_tags(sentence)
#     print(pos_tags)
#     # Convert POS tags to numerical values
#     numerical_tags = [pos_to_num[tag] for tag in pos_tags]
#     # Pad sequences to the same length for FFT
#     # len_numerical_tags = [len(tags) for tags in numerical_tags]
#     # max_length = max(len_numerical_tags)
#     padded_tags = np.pad(numerical_tags, (0, max_length - len(numerical_tags)), mode='constant')
#     print(padded_tags)
#     # Apply FFT
#     fft_result = np.fft.fft(padded_tags)
#     # Use the magnitudes (absolute values) of the FFT result
#     fft_magnitudes = np.abs(fft_result)
#     return fft_magnitudes

def apply_dct_to_pos(sentence, max_length):
    print(f"Sentence: {sentence}")
    pos_tags = get_pos_tags(sentence)
    # print(pos_tags)
    # Convert POS tags to numerical values
    numerical_tags = [pos_to_num[tag] for tag in pos_tags]
    # Pad sequences to the same length for FFT
    # len_numerical_tags = [len(tags) for tags in numerical_tags]
    # max_length = max(len_numerical_tags)
    padded_tags = np.pad(numerical_tags, (0, max_length - len(numerical_tags)), mode='constant')
    
    # print(padded_tags)
    # Apply FFT
    dct_result = dct(padded_tags)
    print(dct_result)
    
    # assert dct_result == dct_type_2(torch.tensor(padded_tags, dtype=torch.float32))
    return dct_result


# # Example: apply to sentences
MAX_LENGTH = max([len(get_pos_tags(sentence)) for sentence in sentences])
dct_features = []
for sentence in sentences:
    dct_feature = apply_dct_to_pos(sentence, MAX_LENGTH)
    dct_features.append(dct_feature)
    print(f"Sentence: {sentence}\nFFT Features: {dct_feature}\n")


for i in range(len(dct_features)):
    for j in range(i+1, len(dct_features)):
        cos = nn.CosineSimilarity(dim=0)
        print(f"Similarity between {sentences[i]} and {sentences[j]} is {cos(torch.tensor(dct_features[i]), torch.tensor(dct_features[j]))}")

# a = np.array([1,2,3,4,5])
# b = np.array([17, 17, 83, 101, 224])
# c = np.array([1,1,1,1,1])

# a = torch.tensor(a, dtype=torch.float32)
# b = torch.tensor(b, dtype=torch.float32)
# c = torch.tensor(c, dtype=torch.float32)

# A = dct.dct(a)
# B = dct.dct(b)
# C = dct.dct(c)
# print(A)
# print(B)
# print(C)
# k = 
# X = dct.dct(x)   # DCT-II done through the last dimension
# y = dct.idct(X)  # scaled DCT-III done through the last dimension
# assert (torch.abs(x - y)).sum() < 1e-10  # x == y within numerical tolerance