from audioop import bias
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import transforms
from lenet_original import LeNet
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

np.random.seed(2)
torch.manual_seed(2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
Step 1:
'''

# MNIST dataset
train_dataset = datasets.MNIST(root='./mnist_data/',
                               train=True, 
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./mnist_data/',
                              train=False, 
                              transform=transforms.ToTensor())

class CombinationConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, weight1, weight2, bias1, bias2, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(CombinationConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # self.alpha = torch.normal(0,1,(1,1)).to(device)
        self.alpha = torch.Tensor([1]).to(device)
        self.alpha = Parameter(self.alpha)
        # self.beta = torch.normal(0,1,(1,1)).to(device)
        self.beta = torch.Tensor([1]).to(device)
        self.beta = Parameter(self.beta)
        self.weight1 = Variable(weight1, requires_grad=False)
        self.weight2 = Variable(weight2, requires_grad=False)
        self.bias1 = Variable(bias1, requires_grad=False)
        self.bias2 = Variable(bias2, requires_grad=False)
        

    def forward(self, input):
        weight_combination = self.alpha * self.weight1 + self.beta * self.weight2
        bias_combination = self.alpha * self.bias1 + self.beta * self.bias2
        bias_combination = bias_combination.view(-1)
        # Perform conv using modified weight.
        return F.conv2d(input, weight_combination, bias_combination, self.stride,
                        self.padding, self.dilation, self.groups)

#     def __repr__(self):
#         s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
#              ', stride={stride}')
#         if self.padding != (0,) * len(self.padding):
#             s += ', padding={padding}'
#         if self.dilation != (1,) * len(self.dilation):
#             s += ', dilation={dilation}'
#         if self.output_padding != (0,) * len(self.output_padding):
#             s += ', output_padding={output_padding}'
#         if self.groups != 1:
#             s += ', groups={groups}'
#         if self.bias is None:
#             s += ', bias=False'
#         s += ')'
#         return s.format(name=self.__class__.__name__, **self.__dict__)

#     def _apply(self, fn):
#         for module in self.children():
#             module._apply(fn)

#         for param in self._parameters.values():
#             if param is not None:
#                 # Variables stored in modules are graph leaves, and we don't
#                 # want to create copy nodes, so we have to unpack the data.
#                 param.data = fn(param.data)
#                 if param._grad is not None:
#                     param._grad.data = fn(param._grad.data)

#         for key, buf in self._buffers.items():
#             if buf is not None:
#                 self._buffers[key] = fn(buf)

#         self.weight.data = fn(self.weight.data)
#         if self.bias is not None and self.bias.data is not None:
#             self.bias.data = fn(self.bias.data)


class CombinationLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, weight1, weight2, bias1, bias2, bias=True):
        super(CombinationLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        

        # weight and bias are no longer Parameters.
        self.weight1 = Variable(weight1, requires_grad=False)
        self.weight2 = Variable(weight2, requires_grad=False)
        self.bias1 = Variable(bias1, requires_grad=False)
        self.bias2 = Variable(bias2, requires_grad=False)

        # self.alpha = torch.normal(0,1,(1,1)).to(device)
        # self.alpha = Parameter(self.alpha)
        # self.beta = torch.normal(0,1,(1,1)).to(device)
        # self.beta = Parameter(self.beta)
        # self.alpha = torch.normal(0,1,(1,1)).to(device)
        self.alpha = torch.Tensor([1]).to(device)
        self.alpha = Parameter(self.alpha)
        # self.beta = torch.normal(0,1,(1,1)).to(device)
        self.beta = torch.Tensor([1]).to(device)
        self.beta = Parameter(self.beta)
    def forward(self, input):
        weight_combination = self.alpha * self.weight1 + self.beta * self.weight2
        bias_combination = self.alpha * self.bias1 + self.beta * self.bias2
        bias_combination = bias_combination.view(-1)
        return F.linear(input, weight_combination, bias_combination)

#     def __repr__(self):
#         return self.__class__.__name__ + '(' \
#             + 'in_features=' + str(self.in_features) \
#             + ', out_features=' + str(self.out_features) + ')'

#     def _apply(self, fn):
#         for module in self.children():
#             module._apply(fn)

#         for param in self._parameters.values():
#             if param is not None:
#                 # Variables stored in modules are graph leaves, and we don't
#                 # want to create copy nodes, so we have to unpack the data.
#                 param.data = fn(param.data)
#                 if param._grad is not None:
#                     param._grad.data = fn(param._grad.data)

#         for key, buf in self._buffers.items():
#             if buf is not None:
#                 self._buffers[key] = fn(buf)

#         self.weight.data = fn(self.weight.data)
#         self.bias.data = fn(self.bias.data)
class LeNet_Combination(nn.Module) :
    def __init__(self, pretrained_model, model1, model2) :
        super(LeNet_Combination, self).__init__()
        model1_param_dict = {}
        model2_param_dict = {}
        for (n1, p1), (n_pre, p_pre) in zip(model1.named_parameters(), pretrained_model.named_parameters()):
                model1_param_dict[n1] = p1 - p_pre
        for (n2, p2), (n_pre, p_pre) in zip(model2.named_parameters(), pretrained_model.named_parameters()):
                model2_param_dict[n2] = p2 - p_pre
        #padding=2 makes 28x28 image into 32x32
        self.C1_layer = nn.Sequential(
                CombinationConv2d(1, 6, 
                                model1_param_dict['C1_layer.0.weight'], model2_param_dict['C1_layer.0.weight'],
                                model1_param_dict['C1_layer.0.bias'], model2_param_dict['C1_layer.0.bias'],
                                kernel_size=5, padding=2),
                nn.Tanh()
                )
        self.P2_layer = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Tanh()
                )
        self.C3_layer = nn.Sequential(
                CombinationConv2d(6, 16, 
                                model1_param_dict['C3_layer.0.weight'], model2_param_dict['C3_layer.0.weight'],
                                model1_param_dict['C3_layer.0.bias'], model2_param_dict['C3_layer.0.bias'],
                                kernel_size=5),
                nn.Tanh()
                )
        self.P4_layer = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Tanh()
                )
        self.C5_layer = nn.Sequential(
                CombinationLinear(5*5*16, 120,
                                model1_param_dict['C5_layer.0.weight'], model2_param_dict['C5_layer.0.weight'],
                                model1_param_dict['C5_layer.0.bias'], model2_param_dict['C5_layer.0.bias'],),
                nn.Tanh()
                )
        self.F6_layer = nn.Sequential(
                CombinationLinear(120, 84,
                                model1_param_dict['F6_layer.0.weight'], model2_param_dict['F6_layer.0.weight'],
                                model1_param_dict['F6_layer.0.bias'], model2_param_dict['F6_layer.0.bias'],),
                nn.Tanh()
                )
        # self.F7_layer = CombinationLinear(84, 10,
        #                         model1_param_dict['F7_layer.weight'], model2_param_dict['F7_layer.weight'],
        #                         model1_param_dict['F7_layer.bias'], model2_param_dict['F7_layer.bias'],)
        self.F7_layer = nn.Linear(84,10)
        self.tanh = nn.Tanh()
        
    def forward(self, x) :
        output = self.C1_layer(x)
        output = self.P2_layer(output)
        output = self.C3_layer(output)
        output = self.P4_layer(output)
        output = output.view(-1,5*5*16)
        output = self.C5_layer(output)
        output = self.F6_layer(output)
        output = self.F7_layer(output)
        return output


'''
Step 3
'''
model1 = LeNet().to(device)
model1.load_state_dict(torch.load('./trained_model/lenet_pmnist0.pt'))
model2 = LeNet().to(device)
model2.load_state_dict(torch.load('./trained_model/lenet_pmnist1.pt'))
pretrained_model = LeNet().to(device)
pretrained_model.load_state_dict(torch.load('./trained_model/lenet_mnist.pt'))
model = LeNet_Combination(pretrained_model, model1, model2).to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

# print total number of trainable parameters
param_ct = sum([p.numel() for p in model.parameters()])
print(f"Total number of trainable parameters: {param_ct}")

'''
Step 4
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)

perm_inds = list(range(28*28))
np.random.shuffle(perm_inds)

import time
start = time.time()
for epoch in range(10) :
    print("{}th epoch starting.".format(epoch))
    for images, labels in train_loader :
        images, labels = images.to(device), labels.to(device)
        images = images.reshape(images.shape[0], -1)
        images = images[:, perm_inds].reshape(images.shape[0], 1, 28, 28)
        optimizer.zero_grad()
        train_loss = loss_function(model(images), labels)
        train_loss.backward()

        optimizer.step()
    print("train_loss : {}".format(train_loss))
end = time.time()
print("Time ellapsed in training is: {}".format(end - start))

torch.save(model.state_dict(), './trained_model/lenet_combination_pretrain_for_pmnist2.pt')

'''
Step 5
'''
test_loss, correct, total = 0, 0, 0

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=100, shuffle=False)

for images, labels in test_loader :
    images, labels = images.to(device), labels.to(device)
    images = images.reshape(images.shape[0], -1)
    images = images[:, perm_inds].reshape(images.shape[0], 1, 28, 28)
    output = model(images)
    test_loss += loss_function(output, labels).item()

    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(labels.view_as(pred)).sum().item()
    
    total += labels.size(0)
            
print('[Test set] Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss /total, correct, total,
        100. * correct / total))

