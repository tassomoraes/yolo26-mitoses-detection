# Detecção de Mitoses com YOLO26

Um pipeline de aprendizado profundo para detecção de figuras mitóticas em imagens histopatológicas de lâminas inteiras, utilizando YOLO26 e outros modelos de detecção de objetos de última geração.

## Visão Geral

Este projeto fornece uma estrutura completa para detecção de mitoses, ou seja, células em divisão, em imagens de patologia digital. Ele oferece suporte a múltiplos conjuntos de dados, incluindo MIDOGpp, MITOS_WSI_CCMCT e MITOS_WSI_CMC, além de incluir recursos de pré-processamento, treinamento, teste e inferência. O sistema utiliza o YOLO26 como modelo principal, mas também oferece suporte a arquiteturas alternativas, como YOLO8, YOLO9, YOLO10, YOLO11, YOLO12 e RT-DETR.

### Vídeo de Apresentação do Projeto
[![Vídeo de Apresentação do Projeto](https://img.youtube.com/vi/b60L-pGy15c/0.jpg)](https://www.youtube.com/watch?v=b60L-pGy15c)

## Funcionalidades

* **Suporte a múltiplos modelos**: YOLO26, como modelo principal, YOLO8, YOLO9, YOLO10, YOLO11, YOLO12, YOLOE, YOLOW e RT-DETR.
* **Suporte a conjuntos de dados**: MIDOGpp, MITOS_WSI_CCMCT, MITOS_WSI_CMC e outros conjuntos de dados de patologia.
* **Pipeline abrangente de pré-processamento**:

  * Extração de patches de imagens de lâmina inteira, ou Whole Slide Images (WSI);
  * Normalização de coloração pelo algoritmo de Macenko;
  * Conversão de coordenadas de anotação para o formato YOLO;
  * Filtragem de patches e controle de qualidade.
* **Recursos avançados de treinamento**:

  * Pipeline de aumento de dados;
  * Suporte a múltiplos otimizadores, como AdamW e SGD;
  * Agendamento da taxa de aprendizado com aquecimento inicial, ou warmup;
  * Regularização, incluindo dropout e weight decay.
* **Utilitários**:

  * Visualização de bounding boxes no formato YOLO;
  * Contagem de anotações e estatísticas;
  * Divisão do conjunto de dados em treino e validação;
  * Ferramentas de processamento em lote.

## Estrutura do Projeto

```txt
yolo26-mitoses-detection/
├── data/                          # Diretórios dos conjuntos de dados
│   ├── MIDOGpp/
│   ├── MITOS_WSI_CCMCT/
│   └── MITOS_WSI_CMC/
├── runs_train/                    # Saídas de treinamento e checkpoints
│   └── yolo26/
│       └── yolo26s/
│           ├── weights/           # Pesos do modelo (best.pt, last.pt etc.)
│           ├── args.yaml
│           └── results.csv
├── src/
│   ├── preprocess/                # Scripts de pré-processamento de dados
│   │   ├── extract_patches.py
│   │   ├── apply_macenko_stain.py
│   │   ├── yolo_label_converter.py
│   │   └── ...
│   ├── train_test/                # Pipelines de treinamento e teste
│   │   ├── main_train.py
│   │   ├── main_test.py
│   │   ├── augmentation_pipeline.py
│   │   ├── constants.py
│   │   └── functions.py
│   └── utils/                     # Funções utilitárias
│       ├── visualize_yolo_boxes.py
│       ├── count_annotation.py
│       └── ...
├── LICENSE                        # Licença MIT
└── README.md
```

## Instalação

### Requisitos

* Python 3.8+
* CUDA 11.0+ para aceleração por GPU
* 8 GB ou mais de RAM
* 50 GB ou mais de espaço em disco para os conjuntos de dados e treinamento

### Configuração

1. Clone o repositório:

```bash
git clone <repository-url>
cd yolo26-mitoses-detection
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

As principais dependências incluem:

* `ultralytics` — implementação do YOLO;
* `torch` — framework de aprendizado profundo PyTorch;
* `torchvision` — utilitários de visão computacional;
* `opencv-python` — processamento de imagens;
* `openslide-python` — manipulação de WSI;
* `numpy`, `pandas`, `scikit-image` — processamento de dados.

## Uso

### 1. Pré-processamento dos Dados

#### Extrair Patches da WSI

Extraia patches de imagens de lâmina inteira e de suas anotações:

```bash
python src/preprocess/extract_patches.py \
  --patch-width 256 \
  --patch-height 256 \
  --svs /path/to/image.svs \
  --db /path/to/annotations.sqlite \
  --output ./patches
```

#### Aplicar Normalização de Coloração

Aplique a normalização de coloração de Macenko às imagens pré-processadas:

```bash
python src/preprocess/apply_macenko_stain.py \
  --input_dir ./patches/images \
  --output_dir ./patches/normalized
```

#### Converter para o Formato YOLO

Converta as anotações para o formato de detecção YOLO:

```bash
python src/preprocess/yolo_label_converter.py \
  --input_dir ./patches/labels \
  --output_dir ./yolo_labels
```

### 2. Treinamento

Configure o conjunto de dados e o modelo em `src/train_test/constants.py`:

```python
MODEL = "yolo26"  # Seleciona a arquitetura do modelo
DATA = "MIDOG_CMC_CCMCT"  # Seleciona o conjunto de dados
```

Execute o treinamento:

```bash
cd src/train_test
python main_train.py
```

**Principais parâmetros de treinamento:**

* `epochs`: 30, por padrão;
* `imgsz`: 640, tamanho da imagem de treinamento;
* `batch`: 16, ajustar conforme a memória da GPU;
* `lr0`: 0.0001, taxa de aprendizado inicial;
* `optimizer`: AdamW;
* `dropout`: 0.1, regularização;
* `weight_decay`: 0.001.

As saídas do treinamento são salvas em `runs_train/yolo26/<model_variant>/`.

### 3. Teste e Inferência

Execute a avaliação no conjunto de teste:

```bash
python src/train_test/main_test.py
```

### 4. Visualização

Visualize as bounding boxes YOLO sobre as imagens:

```bash
python src/utils/visualize_yolo_boxes.py \
  --image_dir ./data/images \
  --label_dir ./data/labels \
  --output_dir ./visualizations
```

### 5. Análise do Conjunto de Dados

Conte as anotações e gere estatísticas:

```bash
python src/utils/count_annotation.py --label_dir ./data/labels
python src/utils/count_images.py --image_dir ./data/images
```

## Configuração do Conjunto de Dados

Cada conjunto de dados requer um arquivo `data.yaml` em seu diretório raiz:

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test

nc: 1  # número de classes (mitoses = 1)
names: ['mitosis']  # nomes das classes
```

## Pesos do Modelo

Os pesos pré-treinados do modelo estão disponíveis em `runs_train/yolo26/yolo26s/weights/`:

* `best.pt` — melhor modelo no conjunto de validação;
* `last.pt` — último checkpoint;
* `epoch0.pt`, `epoch10.pt`, `epoch20.pt` — checkpoints específicos por época.

## Configuração

Edite `src/train_test/constants.py` para personalizar:

* **Seleção do modelo**: escolha entre mais de 10 arquiteturas;
* **Seleção do conjunto de dados**: alterne entre os conjuntos de dados disponíveis;
* **Caminhos**: personalize diretórios de dados e de saída;
* **Parâmetros de treinamento**: ajuste em `main_train.py`.

## Métricas de Desempenho

O pipeline de treinamento gera:

* `results.csv` — métricas por época, como loss, acurácia etc.;
* Matrizes de confusão e curvas de detecção;
* Gráficos de treinamento e validação;
* Checkpoints do modelo em intervalos importantes.

## Licença

Este projeto está licenciado sob a Licença MIT. Consulte o arquivo [LICENSE](LICENSE) para mais detalhes.

## Autor

Tasso Moraes

## Agradecimentos

Este projeto se baseia em:

* [Ultralytics YOLOv8+](https://github.com/ultralytics/ultralytics) — framework central de detecção;
* Conjuntos de dados MIDOG e MITOS — dados de imagem em patologia;
* OpenSlide — biblioteca para manipulação de WSI.

## Contribuição

Contribuições são bem-vindas. Certifique-se de que o código siga as convenções do projeto e inclua a documentação adequada.

## Suporte

Para problemas, dúvidas ou sugestões, abra uma issue no repositório.
