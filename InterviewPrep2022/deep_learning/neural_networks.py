import torch
import torch.nn as nn

# y = W * x + b
# is = classifier = nn.Linear(5,10)

# loss = sum( (y - y_)^2 )

model = nn.Linear(10,3)
loss = nn.MSELoss()

input_vector = torch.randn(10)
target = torch.tensor([0,0,1])

pred = model(input_vector)
output = loss(pred, target)
print(f"Prediction: {pred}")
print(f"Output: {output}")



# Training
def train():
    model = nn.Linear(4,2)

    critertion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        inputs = torch.Tensor([0.8, 0.4,0.4,0.2])
        labels = torch.Tensor([1,0])

        # clear gradient buffers because we don't want to accumulate gradients - carry them into the next iteration
        optimizer.zero_grad()

        # get output of model given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = critertion(outputs, labels)
        print(loss)
        # backpropagate the loss
        loss.backward()
        # update paramters
        optimizer.step()

        print('epoch {}: loss {}'.format(epoch, loss.item()))


# MLP 
model = nn.Sequential(
    nn.Lienar(3,20), # 3 for the input features
    nn.ReLU(),
    nn.Linear(20,2)) # 2 for the classes
    nn.RELU(),
    nn.Linear(2,1))) # 1 for the output

if __name__ == "__main__":
    train()

# PyTorch operations
X= torch.tensor([ 1,2,3,4,5]) 
Y= torch.tensor([[1,2],[3,4]]) 
Z = torch.add(X,Y) 
Z = torch.matmul(X,Y) 
Z = 1 / (1+torch.exp(X)) 


def neuron(inpt):
  weights = torch.tensor([0.5,0.5,0.5])
  bias = torch.tensor([0.5])
  return torch.matmul(inpt, weights) + bias

inpt = torch.tensor([0.2000, 0.4000, 0.2000])
assert neuron(inpt) == torch.tensor([0.9000])
neuron(torch.tensor([0.1000, 0.1000, 0.1000])) == torch.tensor([0.6500])

# Now, it is your turn to build a neural network. Let's have it receive a vector with 10 numbers, have 3 linear layers with dimensions 128, 64, 2, and two RELU layers in between.

seed = 172
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class Model(nn.Module): 
    def __init__(self): 
        super(Model, self).__init__() 
        self.l1 = nn.Linear(10, 128) 
        self.l2 = nn.Linear(128, 64) 
        self.l3 = nn.Linear(64, 2) 
        self.relu = nn.ReLU() 
        self.relu2 = nn.ReLU() 
    def forward(self, x): 
        x = self.l1(x) 
        x = self.relu(x) 
        x = self.l2(x) 
        x = self.relu2(x) 
        x = self.l3(x) 
        return x

def fnn(inpt):
    model = Model()
    Y = model(inpt)
    return Y

fnn(torch.tensor([0.2000, 0.4000, 0.2000, 0.4000, 0.2000, 0.4000, 0.2000, 0.4000, 0.2000,
        0.4000])) == torch.tensor([0.0006, 0.1282])

fnn(torch.tensor([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000,
        0.1000])) == torch.tensor([ 0.0637, -0.0098])