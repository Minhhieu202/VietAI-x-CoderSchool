import torch

torch.manual_seed(2023)


def activation_func(x, func):
    epsilon = 0.01  
    if func == "sigmoid":
        return 1 / (1 + torch.exp(-x))
    elif func == "tanh":
        return torch.tanh(x)
    elif func == "ReLU":
        return torch.maximum(torch.tensor(0.0), x)
    elif func == "leaky_ReLU":
        return torch.where(x > 0, x, epsilon * x)
    else:
        raise ValueError("Hàm kích hoạt không hợp lệ")
    
def softmax(x):
   exp_x = torch.exp(x - torch.max(x))  
   return exp_x / exp_x.sum(dim=1, keepdim=True)

# Define the size of each layer in the network
num_input = 784  # Number of node in input layer (28x28)
num_hidden_1 = 128  # Number of nodes in hidden layer 1
num_hidden_2 = 256  # Number of nodes in hidden layer 2
num_hidden_3 = 128  # Number of nodes in hidden layer 3
num_classes = 10  # Number of nodes in output layer

# Random input
input_data = torch.randn((1, num_input))
# Weights for inputs to hidden layer 1
W1 = torch.randn(num_input, num_hidden_1)
# Weights for hidden layer 1 to hidden layer 2
W2 = torch.randn(num_hidden_1, num_hidden_2)
# Weights for hidden layer 2 to hidden layer 3
W3 = torch.randn(num_hidden_2, num_hidden_3)
# Weights for hidden layer 3 to output layer
W4 = torch.randn(num_hidden_3, num_classes)

# and bias terms for hidden and output layers
B1 = torch.randn((1, num_hidden_1))
B2 = torch.randn((1, num_hidden_2))
B3 = torch.randn((1, num_hidden_3))
B4 = torch.randn((1, num_classes))

hidden_1 = activation_func(torch.matmul(input_data, W1) + B1, func="ReLU")  
hidden_2 = activation_func(torch.matmul(hidden_1, W2) + B2, func="ReLU")  
hidden_3 = activation_func(torch.matmul(hidden_2, W3) + B3, func="ReLU")  
output = torch.matmul(hidden_3, W4) + B4  
result = softmax(output)  

print(result)
print("Tổng :", result.sum().item())

# tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
# Tổng : 1.0