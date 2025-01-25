import torch
import numpy as np
import torch.nn.functional as F

#Create the following tensors:
tensor_3d = torch.zeros(20,30,40)
tensor_1d = torch.arange(10,101,2)

#  x = torch.rand(4, 6)
#     Calculate:
#         1. Sum of all elements of x
#         2. Sum of the columns of x  (result is a 6-element tensor)
#         3. Sum of the rows of x   (result is a 4-element tensor)
x = torch.rand(4, 6)

sum_all = torch.sum(x)
sum_collums = torch.sum(x,dim=0)
sum_rows = torch.sum(x,dim=1)
#  Calculate cosine similarity between 2 1D tensor:
#     x = torch.tensor([0.1, 0.3, 2.3, 0.45])
#     y = torch.tensor([0.13, 0.23, 2.33, 0.45])

x_1d = torch.tensor([0.1, 0.3, 2.3, 0.45])
y_1d = torch.tensor([0.13, 0.23, 2.33, 0.45])
cos_1d = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cosine_similarity_1D = cos_1d(x_1d, y_1d)
# print(cosine_similarity_1D)

# Calculate cosine similarity between 2 2D tensor:

x_2d = torch.tensor([[ 0.2714, 1.1430, 1.3997, 0.8788],
                      [-2.2268, 1.9799, 1.5682, 0.5850],
                      [ 1.2289, 0.5043, -0.1625, 1.1403]])
y_2d = torch.tensor([[-0.3299, 0.6360, -0.2014, 0.5989],
                      [-0.6679, 0.0793, -2.5842, -1.5123],
                      [ 1.1110, -0.1212, 0.0324, 1.1277]])

cos_2d = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
cosine_similarity_2D = cos_2d(x_2d,y_2d)
# print(cosine_similarity_2D)

x = torch.tensor([[ 0,  1],
                      [ 2,  3],
                      [ 4,  5],
                      [ 6,  7],
                      [ 8,  9],
                      [10, 11]])
# Make x become 1D tensor
# Then, make that 1D tensor become 3x4 2D tensor 
x_1d = x.view(-1)
x_2d = x.view(3,4)

# print(x_1d)
# print(x_2d)
# Do the following tasks:
#         1. Make x become 1x3x1080x1920 4D tensor
#         2. Make y become 1x3x720x1280 4D tensor
#         3. Resize y to make it have the same size as x
#         4. Join them to become 2x3x1080x1920 tensor
x_large = torch.rand(3, 1080, 1920)
y_large = torch.rand(3, 720, 1280)

x_large_4d = x_large.unsqueeze(0)
# x_large_4d = x_large_4d.numpy()
# print(type(x_large_4d))
# print(x_large_4d.shape)

y_large_4d = y_large.unsqueeze(0)

y_resize = F.interpolate(y_large_4d, size=(1080 ,1920))
print(y_resize.numpy().shape)
print(x_large_4d.numpy().shape)

join_torch_arr = torch.cat((y_resize, x_large_4d), dim=0)
# print(join_torch_arr.numpy().shape)

# (1, 3, 1080, 1920)
# (1, 3, 1080, 1920)
# (2, 3, 1080, 1920)