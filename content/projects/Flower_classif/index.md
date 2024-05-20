---
title: "Flower Classification using fastAI"
date: 2023-09-01T18:08:42-04:00
draft: false
cover:
    image: img/flor.png
    alt: "Just a moving flower"
    caption: ""
summary: "Program that identifies over a 100 different species of flowers."
tags: ["Python", "Image Processing", "Flowers"]
---

![NeuroRuler demo](img/flor.gif)

## Links

* [GitHub](https://github.com/Mateocontrerass/fastAI-to-make-a-flower-classifier)


## Description

For this project I built a image classifier using the FastAI library to identify over 100 species of flowers. The dataset used for this project can be found [Here](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) and it's part of an Oxford's University repository. 

This is my first real Deep Learning project and I have been following the fastAI methodology of building and then understanding each part and technique.



## Import libraries

```py
# It's always good to have the latest version of fastAI library.
!pip install -Uqq fastai
!pip install timm

import json
import timm
import pandas as pd
import torchvision
import torch
import fastai
import requests

from fastai.vision import *
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, ColorJitter, RandomRotation, RandomResizedCrop, RandomResizedCrop, Compose, ToTensor, Normalize
from fastai.vision.all import *
from fastai import *
from pathlib import Path
from fastai.data.external import untar_data, URLs
```
_Note: It is not a good practice to import libraries using *. However, in this iteration I am aiming to a more interactive approach._

## Load data
I am going to use a fastAI method to download the data from a subdirectory saved in the library using a URL. It has a folder that contains the pictures and three different text files (train, test and valid).
```py
ruta = untar_data(URLs.FLOWERS)
```
_Note: There is no need to download the dataset previously because the former line just did that._
```py
train_txt = pd.read_csv(str(ruta)+'/train.txt', sep =' ', names=['nombre', 'index'])
val_txt = pd.read_csv(str(ruta)+'/valid.txt', sep =' ', names=['nombre', 'index'])
```


The data contains details about the flowers, including the name of the .png files  and a number to indicate the category . However, we need to translate these numbers into names, so we'll be downloading a text file that contains this information, and with the help of a dictionary, we'll rename the numbers to names.

```py
URL = 'https://gist.githubusercontent.com/JosephKJ/94c7728ed1a8e0cd87fe6a029769cde1/raw/403325f5110cb0f3099734c5edb9f457539c77e9/Oxford-102_Flower_dataset_labels.txt'
categorias = pd.read_csv(URL, sep='\t', header=None, names=['Name', 'label'])
categorias['label'] = categorias.index
categorias['Name'] = categorias['Name'].str.replace("'", "")
categorias = categorias.to_dict()['Name']

train_txt["label"] = train_txt['index'].map(categorias)
val_txt["label"] = val_txt['index'].map(categorias)

# Validation mark.
train_txt['is_valid'] =False
val_txt['is_valid']=True

# We concatenate the train and validation datasets.
df = pd.concat([train_txt,val_txt])

df.nombre = df.nombre.apply(lambda x: f'{ruta}/{x}')

```
## Feature engineering

A common problem in machine learning is lacking data, specially using images. That's why fastAI has a couple methods for Data Augmentation such as RandomResizedCrop and aug_transforms, both of which I will be using in this model.

```py
# Se crea el transformador de las imagenes para definir el tamaño de las imagenes que utilizamos
objeto_transf = [RandomResizedCrop(192,min_scale=0.75, ratio=(1.,1.))]

# Se crea el batch_transf para hacerle los siguientes pasos a las imagenes:
# 1. Hacer data augmentation con un tamaño de 192 para aumentar la cantidad de ejemplos en el set
# 2. se normalizan las imagenes para mejorar el rendimiento del modelo segun los stats de imagenet.

batch_transf = [*aug_transforms(mult=2, size=192),Normalize.from_stats(*imagenet_stats)]
```

Once we have all the data set up, we can create the DataBlock.

```py
dblock = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_x = ColReader('nombre'),
    get_y = ColReader('label'),
    item_tfms=objeto_transf,
    batch_tfms=batch_transf,
    splitter=ColSplitter('is_valid')
)

flores = dblock.dataloaders(df)
```

## Training the model

I am going to use a model that already knows how to do something well and train it to make something even better for our needs. This is called transfer learning and it allows us to train models faster. 

The pretrained model I am going to use is called **convnext_tiny_in22k** (previously i tried a **resnet50** variation but i found out that the model I am currently using has a better performance).

```py
model = timm.create_model('convnext_tiny_in22k', pretrained=True, num_classes=flores.c)

learner_2 = Learner(flores, model, metrics= error_rate)
learner_2.fine_tune(4)
```

## Saving the model
The next line will download the model as a __.pkl__ file which can be used to deploy the model.
```py
learner_2.export()
```
## Model deployment 
To deploy the model I am going to use Hugging Face Spaces and Gradio using the following script. It will require to upload the __.pkl__ file and an optional example file.

```py
from fastai.vision.all import *
import gradio as gr
import timm


learner = load_learner("./export.pkl")


categorias = learner.dls.vocab

def clasificar_imagen(img):
    prediccion, indice, probabilidades = learner.predict(img)
    return dict(zip(categorias,map(float,probabilidades)))

imagen = gr.inputs.Image(shape=(500,500))
etiqueta = gr.outputs.Label()

ejemplo = ['girasol.jpg']

interfaz = gr.Interface(fn=clasificar_imagen,inputs=imagen,outputs=etiqueta, examples=ejemplo)
interfaz.launch(inline=False)

```

## Try it out
Check the result and let me know how it goes! __:)__
{{< rawhtml >}}

<iframe
    src="https://mateocontreras-fastai-flower-classif.hf.space"
    frameborder="0"
    width="850"
    height="450"
></iframe>


{{< /rawhtml >}}

