# Detecção de Imagens Falsas com GAN

Este projeto utiliza uma GAN (Rede Adversarial Generativa) para treinar um discriminador capaz de classificar imagens como reais ou falsas. Também inclui uma interface interativa via Jupyter Notebook, onde o usuário pode testar imagens do dataset, imagens geradas ou carregar sua própria imagem para avaliação.

---

## Requisitos

Antes de tudo, certifique-se de ter o Python 3.8 ou superior instalado.

### Instale as bibliotecas necessárias:

```bash
pip install tensorflow matplotlib ipywidgets pillow
```

### Estrutura esperada do projeto

```
projeto_pdi/
│
├── dataset/
│   └── valid/
│       ├── real/
│       │   ├── imagem1.jpg
│       │   └── ...
│       └── fake/
│           ├── imagem1.jpg
│           └── ...
│
├── projeto_gan.ipynb
└── README.md
```

### Etapas de Execução
1 - Abra o Jupyter Notebook (recomendado dentro do VS Code).

2 - Execute todas as células do notebook na ordem.

3 - O modelo será treinado por algumas épocas (configurável).

4 - Após o treinamento, a interface interativa estará disponível.


### O que cada parte faz?
1 - Carregamento e normalização do dataset com imagens 64x64.

2 - Definição do discriminador (classifica real/fake).

3 - Definição do gerador (cria imagens falsas a partir de ruído).

4 - Criação da GAN combinando gerador + discriminador.

5 - Loop de treinamento com acompanhamento de perdas.

6 - Interface interativa com widgets para teste de imagens.
