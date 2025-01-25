import torch

torch.manual_seed(2023)


def activation_func(x, func = "ReLU"):
    #TODO Implement one of these following activation function: sigmoid, tanh, ReLU, leaky ReLU
    epsilon = 0.01   # Only use this variable if you choose Leaky ReLU
    if func == "sigmoid":
        return 1/(1+torch.exp(-x))
    # elif func == "tanh":
    #     return torch.tanh(x)
    # elif func == "ReLU":
    #     return torch.maximum()
    else:
        raise ValueError ("ham kich hoat khong dung")
   
def softmax(x):
    exp_x = torch.exp(x - torch.max(x))
    return exp_x / exp_x.sum(keepdim=True)


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

result = None
print(result)

# tensor([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
# Tổng : 1.0