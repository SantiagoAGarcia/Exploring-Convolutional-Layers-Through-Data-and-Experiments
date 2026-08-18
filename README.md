# Convolutional Layers Workshop

Clothing image classification (Fashion-MNIST) comparing a dense network with a
convolutional network, plus model training and deployment on Amazon SageMaker.

The goal is not to achieve the best possible accuracy, but to understand what a convolutional
layer contributes. The question I am trying to answer is whether the CNN wins because it has more
capacity, or because it is structured in a way that better fits images.

## Repository structure

```
.
├── README.md
├── cnn_workshop.ipynb              # main work (EDA, baseline, CNN, experiment, interpretation)
├── sagemaker_train_deploy.ipynb    # SageMaker training and endpoint deployment
├── entrenar_en_docker.py           # reproduces SageMaker training with Docker, without AWS
├── requirements.txt
├── sagemaker/
│   ├── train.py                    # training script that runs inside the container
│   └── inference.py                # input preparation and endpoint response
└── artifacts/
    └── resultados.json             # complete metrics (generated when the notebook is run)
```

`artifacts/` also receives the exported model when the notebook is run; you do not need to
createla manually; the notebook generates it.

To run it:

```bash
pip install numpy matplotlib tensorflow
jupyter notebook cnn_workshop.ipynb
```

There is a `MODO_RAPIDO` flag in the configuration cell. With `True`, it trains on less
data to verify that everything works; the results here were run with `False`.

---

## The problem

Classify 28x28 grayscale images into 10 clothing categories. The input is a
tensor `(28, 28, 1)`, the output is 10 logits, and the loss is
`SparseCategoricalCrossentropy(from_logits=True)`.

## The dataset

Fashion-MNIST, loaded directly from `tf.keras.datasets`.

| | |
|---|---|
| Images | 60,000 training, 10,000 test |
| Split used | 54,000 train / 6,000 validation / 10,000 test |
| Size | 28 x 28, un solo canal |
| Classes | 10, with 6,000 images each |
| Values | integers from 0 to 255 |

**Why it is suitable for convolutions.** Convolutional layers assume that what
matters is found in small regions of the image, and that a useful detail in one part is also
useful in another. Both assumptions hold for clothing: what separates a boot from a sandal
is the shaft height or the gaps between straps, not the relationship between two
opposite pixels.

To avoid relying only on that, in section 2.4 of the notebook I compare how similar
nearby pixels are compared with distant pixels. Neighbors are much more similar, and similarity
drops quickly with distance, so looking only at small neighborhoods is not leaving out
important information.

**Why not MNIST.** It was the obvious choice because we had already used it in class, but there a simple dense network
already exceeds 97%, leaving very little room for an improvement to stand out.
Fashion-MNIST has the same format and is harder, especially because several classes
look similar to each other.

**Preprocessing:** divide by 255, add the channel dimension, and set aside 10% for
validation. Nothing else. I do not resize (they are already the same size) or augment data, so that
the only difference between the models is the architecture. I use the test set
only once per model, at the end.

---

## The models

### Baseline model (no convolutions)

```
entrada (28,28,1) -> Flatten(784) -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, linear)
```

**109,386 parameters.**

The important point here is `Flatten`: when flattening, the top-left pixel and the
bottom-right pixel become two arbitrary positions in a vector. The network loses
any notion of which pixels were neighbors before the first layer.

At first I made this model smaller, but then the comparison was not useful: if the
CNN wins with twice as many parameters, I cannot tell whether it won because of convolution or because it is
larger. I deliberately made it similar in size to the CNN.

### CNN

```
entrada (28,28,1)
  Conv2D(16, 3x3, same, relu)  -> (28,28,16)      160 par.
  MaxPooling2D(2x2)            -> (14,14,16)
  Conv2D(32, 3x3, same, relu)  -> (14,14,32)    4.640 par.
  MaxPooling2D(2x2)            -> (7,7,32)
  Flatten                      -> 1568
  Dense(64, relu)                             100.416 par.
  Dense(10, linear)                               650 par.
```

**105,866 parameters.**

Something I did not expect when doing the calculations: the two convolutional layers together have 4,800
parameters, less than 5% of the total. Almost the entire model is the final dense layer, and yet
those 4,800 are what make the difference.

### Why each decision

| Decision | Chose | Why |
|---|---|---|
| conv layers | 2 | with 2 layers and their pooling, each final neuron sees a 10x10 region over 28x28, enough to cover part of the garment. With 1 the window is too small and with 4 the map shrinks too quickly |
| kernel | 3x3 | pixels stop resembling each other after a few pixels, so a small window should be enough. Also, two 3x3 layers see the same region as one 5x5 layer with fewer parameters. This is what I test in the experiment |
| filters | 16 → 32 | I double them when pooling halves the map, so as not to lose too much information at once. I start at 16 because the first layer only detects edges |
| stride | 1 in conv, 2 in pooling | with stride 1 the convolution does not skip anything; I leave image reduction to pooling, which has no parameters |
| padding | `same` | without padding each conv crops the border, where there may still be clothing. It also makes the output size independent of the kernel, which is useful for the experiment |
| activation | ReLU | with so much black background many activations will be zero anyway, and ReLU does not saturate for positive values |
| output | `linear` | goes with `from_logits=True`, which is the stable approach. I apply softmax afterward |
| pooling | MaxPooling 2x2 | achica el cómputo, agranda la zona que ve cada neurona sin agregar parámetros y hace que un corrimiento chico no cambie el resultado. Chose max en vez de average porque me interesa si el patrón aparece, y con tanto fondo negro el promedio lo diluiría |

### Compact CNN (control)

The two models above share the same 100,000-parameter dense layer, so I wondered
how much of the improvement comes from convolution and how much from that layer. I built
a third version that replaces `Flatten + Dense(64)` with `GlobalAveragePooling2D`:

```
Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> GlobalAvgPool -> Dense(10)
```

**23,946 parameters**, almost 5 times fewer than the baseline. The idea was: if it still gets close,
then the advantage is not a matter of size.

**It did not work.** It reached 0.8266 versus 0.8769 for the baseline, five points lower. But
it ended with 0.8609 training and 0.8377 validation accuracy, both still increasing in la
última época y sin nada de sobreajuste, mientras las otras dos redes ya se habían aplanado.
So it was undertrained: `GlobalAveragePooling2D` abruptly discards all the
información de dónde estaba cada activation y la red tarda más en compensarlo.

So this control **does not answer the question I wanted to answer**, and I leave it noted
as such rather than forcing a conclusion. The main comparison (baseline versus CNN
with roughly the same number of parameters) does not depend on it.

---

## Experiment: kernel size

I tested 3x3 versus 5x5 versus 7x7, keeping everything else fixed (2 conv layers, 16 y 32
filters, stride 1, padding `same`, pooling, ReLU, la misma capa densa, Adam con lr = 1e-3,
batch 128, mismas épocas, misma partición).

Two design details are worth clarifying:

**The `same` padding is intentional.** Hace que el tamaño de output no dependa del kernel,
so the vector entering the dense layer is always 1568 y esa capa tiene los mismos
parámetros en las tres pruebas. Lo único que cambia son los parámetros de las
convoluciones. Si no fuera así estaría cambiando dos cosas a la vez.

**I run each configuration 3 times.** La primera vez entrené una sola vez cada kernel y me
dio que 5x5 era el mejor; después volví a correr lo mismo y me dio distinto. Como los
pesos arrancan al azar, con una sola corrida no se puede distinguir una diferencia real de
la casualidad.

### Cost of each kernel

| Kernel | Conv parameters | Total parameters | Region seen by each neuron |
|---|---:|---:|---:|
| 3x3 | 4.800 | 105.866 | 10 x 10 |
| 5x5 | 13.248 | 114.314 | 16 x 16 |
| 7x7 | 25.920 | 126.986 | 22 x 22 |

### Results

Run for 15 epochs, batch 128, Adam con lr = 1e-3. The complete numbers are in
`artifacts/resultados.json`.

| Modelo | Parámetros | Train acc | Val acc | Test acc | Segundos |
|---|---:|---:|---:|---:|---:|
| Modelo base denso | 109.386 | 0.9234 | 0.8890 | 0.8769 | 13 |
| CNN | 105.866 | 0.9452 | 0.8993 | **0.9006** | 53 |
| CNN compacta | 23.946 | 0.8609 | 0.8377 | 0.8266 | 66 |

| Kernel | Val acc (promedio de 3 semillas) | Variación entre corridas | Brecha train-val | Segundos |
|---|---:|---:|---:|---:|
| 3x3 | 0.9057 | 0.0035 | +0.0240 | 38 |
| 5x5 | 0.9045 | 0.0052 | +0.0290 | 50 |
| 7x7 | 0.9033 | 0.0061 | +0.0308 | 78 |

### Analysis

**In accuracy, the three kernels are tied.** La diferencia entre el mejor y el peor es
0.0024, and runs of the same kernel vary by up to 0.0061 entre sí solo por cómo
arrancaron los pesos. The difference is smaller than the noise, así que no se puede afirmar
que un kernel sea mejor que otro.

This is where running three seeds helped: con una sola habría visto que 3x3 quedó primero
y habría concluido que gana, when that difference means nothing.

**Where there is a clear difference is cost.** Going from 3x3 to 7x7 multiplies
the convolution parameters by 5.4 and doubles the time, for the same result. Y la
brecha entre entrenamiento y validación crece de +0.0240 a +0.0308, o sea que los kernels
grandes se sobreajustan un poco más.

**My hypothesis was supported, but in a different way.** I expected 3x3 to be enough. No
es que 3x3 gane: es que agrandar el kernel no aporta nada mientras cuesta cinco veces más.
When two options give the same result, choose the cheaper one.

This matches what I saw when measuring pixel correlation: a 5 píxeles
todavía quedaba 0.41, so it was reasonable to expect 5x5 not to be worse. Lo que no le alcanza es
para ser mejor.

| Kernel | Pros | Cons |
|---|---|---|
| 3x3 | the same result with 5 times fewer parameters and half the time | sees little per layer; it needs stacking |
| 5x5 | sees a larger region at once | 2.8 times more parameters for the same result |
| 7x7 | a lot of immediate context | 5.4 times more parameters, twice the time, slightly more overfitting, and the same result |

**Conclusión:** conviene 3x3, no porque dé mejor exactitud sino porque da la misma mucho
más barato. Si hiciera falta ver una zona más grande, lo sensato sería apilar capas chicas
antes que agrandar el kernel.

---

## Interpretation

### An extra test: shifting the images

Se me ocurrió mientras escribía el análisis. Corrí las imágenes de prueba unos píxeles,
rellenando con negro, y evalué los dos modelos sin volver a entrenarlos.

| Shift | Baseline | CNN | CNN advantage |
|---|---:|---:|---:|
| 0 px | 0.8769 | 0.9006 | +0.024 |
| 1 px | 0.7452 | 0.8357 | +0.091 |
| 2 px | 0.4273 | 0.6623 | **+0.235** |
| 3 px | 0.2265 | 0.3030 | +0.077 |
| 4 px | 0.1458 | 0.1733 | +0.028 |

The result has two parts. **With small shifts, the CNN holds up much better**: a
2 píxeles el denso se cayó a 0.43 y la CNN sigue en 0.66, 23 puntos de diferencia. **Pero a
3 y 4 píxeles se caen las dos**, hasta quedar por debajo del 20%, que con 10 clases es
apenas mejor que tirar una moneda.

The tolerance comes from the two `MaxPooling2D` layers de 2x2, que dan margen de unos pocos píxeles,
y justo hasta ahí llega la ventaja. Más allá no hay nada que la sostenga, porque la CNN
termina en un `Flatten` seguido de una capa densa, y esa capa sí depende de en qué posición
apareció cada activation.

This clarified something I had misunderstood: convolution does not provide translation invariance for free.
Da que el mismo filtro detecte el mismo patrón en cualquier lado, que no es lo mismo. Si
después aplanás y conectás a una densa, la posición vuelve a importar. Los supuestos no
vienen solo del tipo de capa, vienen de cómo armás la red entera.

### ¿Why la CNN funcionó mejor?

La CNN llegó a 0.9006 contra 0.8769 del modelo denso: dos puntos y medio, con 3.520
parámetros menos. Of the three arguments I wanted to use, two hold up and one does not:

1. **Con la misma cantidad de parámetros la CNN gana** (105.866 contra 109.386). Si solo
   importara el tamaño, deberían haber empatado. 
2. **La versión compacta no le llegó al modelo base.** Quedó a medio entrenar, así que no
   sirve como argumento ni a favor ni en contra. 
3. **La CNN aguanta mejor los corrimientos chicos**, con la salvedad de arriba.

**Where exactly it wins.** Las mejoras están en pullover (+9,3 puntos), camiseta (+7,3) y
camisa (+2,9), justo las prendas de torso que comparten silueta y se distinguen por
textura. En las clases fáciles (pantalón, botín, zapatilla) mejora medio punto o menos
porque no había margen, y en abrigo queda dos puntos **peor**.

That is the most interesting part of the result: la CNN no mejora parejo, mejora justo donde hacía
falta mirar el detalle fino, que era la predicción que había hecho mirando las imágenes
antes de entrenar.

Lo que creo que pasa por debajo: un filtro de 3x3 se aplica en las 784 posiciones de la
imagen, así que sus 9 pesos reciben información de todas esas posiciones en cada imagen. Un
peso de la capa densa solo ve un píxel por imagen.

**Honest caveat:** dos puntos y medio no es espectacular, por dos motivos. Las
imágenes son chicas y están centradas, así que el problema de la posición variable casi no
existe. Y el 95% de los parámetros de la CNN están en la capa densa final: en el fondo son
dos redes que comparten casi toda la estructura salvo el principio.

### What assumptions does convolution bring?

Al elegir la arquitectura le metemos al modelo ideas nuestras sobre los datos antes de que
vea un ejemplo. En la convolución son cuatro:

| Assumption | How it is built into the network | What I saw in my tests |
|---|---|---|
| what matters is local | cada neurona se conecta solo a una ventana de `k x k` | holds: la correlación cae a cero a los 10 px |
| what works here works there | el mismo filtro recorre toda la imagen | holds: gana con menos parámetros |
| moving the image does not change what it is | sale de repetir los mismos pesos | **only partly**: aguanta 1-2 px, no 4 |
| complex things are built from simple ones | apilar capas hace que cada una vea más zona | los filters aprendidos son detectores de borde |

What struck me most es que todos son **restricciones**: una capa convolucional
puede representar menos cosas que una densa del mismo tamaño, no más. Y sin embargo
funciona mejor.

La forma en que lo entendí: la red densa tiene que buscar la solución entre muchísimas
opciones, y la mayoría no tienen ningún sentido para una imagen (por ejemplo tratar dos
píxeles lejanos como si estuvieran relacionados). La CNN tiene prohibidas esas opciones de
entrada, así que busca en un conjunto mucho más chico donde casi todo es razonable. Por
eso necesita menos datos.

Es parecido a lo que hacíamos en el primer notebook del curso armando características a
mano, solo que en vez de decidir qué características usar, decidimos las reglas con las
que la red las aprende sola.

### When is it not a good fit?

Cuando esos supuestos no holdsn:

- **Tabular data.** El orden de las columnas es arbitrario: no hay motivo para que edad
  e ingresos estén "al lado" ni para pasar el mismo filtro sobre columnas que miden cosas
  distintas. Mejor una red densa o árboles.
- **When exact position matters.** Si la tarea es "¿hay algo brillante arriba a la
  izquierda?", que la red responda igual sin importar dónde aparece el patrón es
  justamente lo que no quiero. Pasa en imágenes médicas donde la posición anatómica es
  parte del diagnóstico.
- **When what matters is far apart.** En texto el sujeto y el verbo pueden estar
  separados por muchas palabras; una CNN necesitaría muchísimas capas para llegar. Para
  eso están los mecanismos de atención.
- **When the data does not form a grid.** Grafos, redes sociales, nubes de puntos: no
  existe "el de al lado a la derecha".

The question I take away, más útil que "si son imágenes usá CNN", es: **¿los datos tienen
vecinos que significan algo, y un patrón que sirve en un lugar sirve también en otro?** Si
las dos respuestas son sí, la convolución ayuda. Si alguna es no, esos supuestos se
vuelven una limitación.

---

## Learned filters (bonus)

Section 7 of the notebook shows los pesos de la primera capa y los mapas de activation.
Los filters no son ruido: se ven patrones con una parte clara y otra oscura en distintas
direcciones, que es la pinta de un detector de bordes, y nadie los programó. En los mapas
de `conv_1` todavía se reconoce la silueta de la prenda; en `conv_2` ya son más chicos y
más difíciles de interpretar.

---

## SageMaker deployment

```
datos (.npz)
      |  file://  en modo local, o subida a S3 en modo nube
      v
 entrenamiento  ->  contenedor TensorFlow  ->  sagemaker/train.py
      |
      v
 model.tar.gz
      |  deploy
      v
 endpoint  ->  TensorFlow Serving + sagemaker/inference.py
      |
      v
 respuesta JSON con la clase, la etiqueta y la confianza
```

### How it was run: containerized training without AWS

I ran out of course credits, así que no pude usar ni las instancias de SageMaker ni su
Local Mode (que igual necesita credenciales de AWS para bajarse la imagen del contenedor
desde ECR).

What I did was reproduce the setup con la imagen pública `tensorflow/tensorflow:2.13.0`
de Docker Hub, que no pide autenticación:

```bash
python entrenar_en_docker.py --epochs 15
```

SageMaker script mode does nothing special: mete tu `train.py` en un contenedor, monta
los datos en `/opt/ml/input/data/{train,validation}`, define las variables `SM_MODEL_DIR`,
`SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION` y `SM_OUTPUT_DATA_DIR`, y lo ejecuta. El script
[`entrenar_en_docker.py`](entrenar_en_docker.py) arma exactamente ese contenedor.

**`sagemaker/train.py` was not modified by a single line**: es el mismo archivo que usaría un
training job real. El contenedor deja el `SavedModel` en `output_docker/model/1`, igual que
SageMaker lo dejaría en `/opt/ml/model`.

Result of that run: `val_accuracy` 0.9080, y cargando el `SavedModel` que produjo el
contenedor da **0.919** sobre 2.000 imágenes de prueba.

**What this is not:** a SageMaker endpoint. Eso requiere una cuenta con créditos. Lo
que sí demuestra es que el script de entrenamiento funciona containerizado y que el modelo
exportado es válido y se puede servir.

The notebook [`sagemaker_train_deploy.ipynb`](sagemaker_train_deploy.ipynb) queda completo y
listo para correr en cuanto haya créditos: tiene la bandera `MODO_LOCAL` para alternar
entre Docker y las instancias de AWS.

> Si instalás el SDK, fijalo a `sagemaker>=2.190,<3`. La versión 3 eliminó `sagemaker.local`,
> `sagemaker.tensorflow`, `Session` y `get_execution_role`, así que el notebook falla en la
> primera celda.

The notebook tiene una bandera para pasar de uno a otro:

| | `MODO_LOCAL = True` | `MODO_LOCAL = False` |
|---|---|---|
| Where it trains | Docker container | `ml.m5.xlarge` instance |
| Endpoint | local container on a port | SageMaker endpoint |
| Data | local folder with `file://` | uploaded to S3 |
| Costo | uses no credits | charged by the hour |

To move it to the cloud, changing that variable is enough: lo único que cambia por debajo es
el `instance_type` y de dónde se leen los datos.

**To run it locally you need:** Docker Desktop abierto, `pip install
"sagemaker[local]"` y credenciales de AWS configuradas. Lo último me sorprendió: aunque
todo se ejecute en mi máquina, el SDK igual necesita autenticarse para bajarse la imagen
del contenedor la primera vez. Bajar la imagen no consume créditos de cómputo y después
queda en caché.

The first run takes a while porque la imagen de TensorFlow pesa varios gigas.

### Deployment details

| Stage | Detalle |
|---|---|
| Data | un `.npz` por canal |
| Training | contenedor TensorFlow 2.13, modo script |
| Code | `sagemaker/train.py`, con la misma arquitectura del notebook |
| Metrics | salen de los logs con `metric_definitions`; en la nube quedan en CloudWatch |
| Endpoint | TensorFlow Serving |

In cloud mode I would use CPU rather than GPU, porque con 106.000 parámetros e imágenes de 28x28 la GPU
no se aprovecha.

The decision I find most interesting es dónde poner el softmax. El modelo devuelve
logits porque se entrenó con `from_logits=True`, que es la forma estable. Si lo hubiera
metido adentro del modelo tendría que haber cambiado la pérdida. Dejándolo en
`output_handler` me quedan las dos cosas: entrenamiento estable y una respuesta que se
entiende del otro lado.

**Note if running in the cloud:** el endpoint charged by the hour mientras exista. La última
celda del notebook lo apaga, conviene correrla siempre al terminar.

---

## What remains to be done

First are the two things that turned out differently than expected and remain unresolved:

- **Train the compact CNN until it converges.** Es lo que más me interesa, porque es el
  control que quedó sin responder. Terminó todavía subiendo.
- **Measure the compact CNN's robustness to shifts.** Como usa
  `GlobalAveragePooling2D` en vez de `Flatten`, debería aguantar mucho mejor los
  corrimientos grandes. Es una predicción concreta que sale del análisis y que no llegué a
  probar.

And then:

- I tested only one small-image dataset. No sé si lo del kernel se mantendría con
  imágenes más grandes; tengo entendido que ahí sí se usan kernels más grandes en las
  primeras capas, pero no llegué a probarlo.
- Three runs per configuration are useful para ver si la diferencia supera al ruido, pero
  no es una prueba estadística en serio.
- I deliberately did not use data augmentation, dropout, or batch normalization, para aislar el
  efecto de la arquitectura. Me quedó la duda de cuánto mejoraría el modelo denso con algo
  de regularización, porque es el que más se sobreajusta.
- I could not interpret what each filter detects in the second-layer maps. qué detecta cada filtro.

**What I would try next:** entrenar el modelo denso con las imágenes corridas al azar,
para ver si aprende solo a aguantar los corrimientos o si de verdad hace falta que la
arquitectura se lo imponga.
