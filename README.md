# Classificação de Dígitos MNIST com PyTorch

Este repositório contém uma implementação de uma Rede Neural Convolucional (CNN) em PyTorch para classificar dígitos escritos à mão do conjunto de dados MNIST. O MNIST é um conjunto de dados amplamente utilizado em visão computacional, composto por imagens em escala de cinza de dígitos manuscritos de 0 a 9.

## Arquivos Principais

- `mnistDigitClassification.py`: Este é o arquivo principal em Python que contém o código para definição da CNN, treinamento e teste do modelo.

## Pré-requisitos

Antes de executar o código, certifique-se de ter instalado as seguintes bibliotecas Python:

- PyTorch
- torchvision

Você pode instalá-las usando o comando pip:

`pip install torch torchvision`


## Descrição do Código

### Definindo a Arquitetura da CNN

A arquitetura da CNN é definida na classe `Net`:

- Duas camadas de convolução 2D com ativação ReLU.
- Camadas de dropout para evitar overfitting.
- Duas camadas totalmente conectadas para a classificação.

### Carregando o Conjunto de Dados MNIST

Utilizamos a biblioteca torchvision para carregar o conjunto de dados MNIST e aplicamos transformações para normalizar as imagens.

### Inicializando a CNN e Definindo a Função de Perda e Otimizador

Inicializamos a CNN, definimos a função de perda como a entropia cruzada (`nn.CrossEntropyLoss`) e escolhemos o otimizador Adam para treinar o modelo.

### Treinando a CNN

Treinamos o modelo usando os dados de treinamento por várias épocas. A cada época, as imagens são passadas pela rede, a perda é calculada e o modelo é otimizado usando o algoritmo de retropropagação.

### Salvando o Modelo Treinado

Após o treinamento, salvamos os pesos do modelo em um arquivo chamado 'modelo_mnist.pth'.

### Testando a CNN

Colocamos o modelo em modo de avaliação e testamos sua precisão em um conjunto de dados de teste. A precisão é calculada como a porcentagem de previsões corretas em relação ao total de imagens de teste.

## Executando o Código

Para executar o código, basta executar o arquivo `mnistDigitClassification.py`. Certifique-se de ter instalado as dependências mencionadas anteriormente.

`python mnistDigitClassification.py`


## Resultados

Ao final da execução, você verá a precisão do modelo na classificação dos dígitos MNIST no conjunto de teste.
