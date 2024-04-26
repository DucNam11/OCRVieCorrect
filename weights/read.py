import torch

# Đọc file .pth
checkpoint = torch.load('C:\\Users\\Admin\\Desktop\\OCRVieCorrect\\weights\\seq2seq_0.pth', map_location=torch.device('cpu'))


# In ra vài trọng số quan trọng từ encoder
print("Trọng số của lớp nhúng của encoder:")
print(checkpoint['encoder.embedding.weight'])

# In ra vài trọng số quan trọng từ decoder
print("Trọng số của lớp nhúng của decoder:")
print(checkpoint['decoder.embedding.weight'])

# In ra vài trọng số quan trọng từ lớp attention của decoder
print("Trọng số của lớp attention của decoder:")
print(checkpoint['decoder.attention.attn.weight'])

# In ra vài trọng số quan trọng từ lớp GRU của decoder
print("Trọng số của lớp GRU của decoder:")
print(checkpoint['decoder.gru.weight_ih_l0'])
