# Imports necessários
# *****************************************************************************
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# Transformações nas imagens para processamento:
# Transformar para tensor
# Normalizar os canais RGB com uma média e desvio padrão de 0.5
# *****************************************************************************
data_transforms = {
    'treinamento': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'teste': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}

# Caminho das pastas de imagens para treinamento e teste
# Mude este caminho de acordo o seu sistema!!!
# *****************************************************************************
data_dir = 'D:\Documentos\Vale - Hub Inovação\ImagensCNNBuracos'

# Criação dos conjuntos de dados
# *****************************************************************************
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['treinamento', 'teste']}

# Agrupamento das imagens em um dataloader para processamento
# Pense nisso como um container com as imagens agrupadas em lotes
# *****************************************************************************
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=5,
                                             shuffle=True, num_workers=1)
              for x in ['treinamento', 'teste']}

# Informações úteis dos conjuntos para serem mostradas no terminal
# *****************************************************************************
dataset_sizes = {x: len(image_datasets[x]) for x in ['treinamento', 'teste']}
print(dataset_sizes)
class_names = image_datasets['treinamento'].classes
print(class_names)

print('Quantidade de imagens para treinamento: ', dataset_sizes['treinamento'])
print('Quantidade de imagens para teste: ', dataset_sizes['teste'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Device: {device}')


# Função para visualizar algumas imagens
# *****************************************************************************
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(5)  # pause a bit so that plots are updated


# Busca um batch de imagens do grupo treinamento
# As informações das imagens e das respectivas legendas são carregadas nas variáveis
inputs, classes = next(iter(dataloaders['treinamento']))

# Criação de um grid a partir do batch obtido acima
out = torchvision.utils.make_grid(inputs, nrow=5)

# Mostra as imagens do batch usando a função imshow()
imshow(out, title=[class_names[x] for x in classes])


# Classe que define a estrutura da rede CNN com as camadas de convolução 
# e classificação definidas de acordo a nossa escolha
# *****************************************************************************
class Net(nn.Module):
    
    # Função inicial que define as camadas da rede
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    # Função que define a passagem dos dados pela rede
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Criação do objeto da rede
# *****************************************************************************
net = Net()

# Criação dos critérios de treinamento: Função de erro e otimizador
# Erro: CrossEntropyLoss
# Otimizados: stochastic gradient descent
# *****************************************************************************
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# Loop de treinamento da rede 
# (Poder ser encapsulado em uma função!)
# *****************************************************************************
for epoch in range(50):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(dataloaders['treinamento'], 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 5 == 0:    # print every 5 mini-batches
            print(f'[Epoch: {epoch + 1}, Batch: {i + 1}] loss: {running_loss / 5 :.5f}')
            running_loss = 0.0

print('Finished Training')

# Etapa para salvar o modelo da rede treinanda
# Mude este caminho de acordo o seu sistema!!!
# *****************************************************************************
PATH = 'D:\Documentos\Vale - Hub Inovação\ImagensCNNBuracos/rede.pth'
torch.save(net.state_dict(), PATH)

# Utilização da função imshow() para mostrar um batch do conjunto de teste
# *****************************************************************************
dataiter = iter(dataloaders['teste'])
images, labels = dataiter.next()

# Mostra as legendas das imagens de teste
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % class_names[labels[j]] for j in range(5)))

# Carregamento da rede salva para classificar as imagens do batch de teste
# *****************************************************************************
net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)

_, predicted = torch.max(outputs, 1)

# Mostra o resultado das classificações das imagens do conjunto de teste
print('Predicted: ', ' '.join('%5s' % class_names[predicted[j]]
                              for j in range(5)))

# Loop para classificar todas as imagens de teste e verificar a taxa de acerto
# *****************************************************************************
correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['teste']:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Taxa de classificações corretas para as 60 imagens de teste: %d %%' % (
    100 * correct / total))