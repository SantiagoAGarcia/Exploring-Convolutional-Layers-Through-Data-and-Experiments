# Workshop de capas convolucionales

Clasificación de imágenes de ropa (Fashion-MNIST) comparando una red densa con una red
convolucional, más el entrenamiento y despliegue del modelo en Amazon SageMaker.

La idea no es sacar la mejor exactitud posible sino entender qué aporta una capa
convolucional. La pregunta que trato de responder es si la CNN gana porque tiene más
capacidad o porque está armada de una forma que encaja mejor con las imágenes.

## Estructura del repositorio

```
.
├── README.md
├── cnn_workshop.ipynb              # trabajo principal (EDA, modelo base, CNN, experimento, interpretación)
├── sagemaker_train_deploy.ipynb    # entrenamiento en SageMaker y despliegue en un endpoint
├── entrenar_en_docker.py           # reproduce el entrenamiento de SageMaker con Docker, sin AWS
├── requirements.txt
├── sagemaker/
│   ├── train.py                    # script de entrenamiento que corre dentro del contenedor
│   └── inference.py                # preparación de la entrada y de la respuesta del endpoint
└── artifacts/
    └── resultados.json             # métricas completas (se genera al correr el notebook)
```

`artifacts/` también recibe el modelo exportado al correr el notebook; no hace falta
crearla a mano, el propio notebook la genera.

Para correrlo:

```bash
pip install numpy matplotlib tensorflow
jupyter notebook cnn_workshop.ipynb
```

Hay una bandera `MODO_RAPIDO` en la celda de configuración. En `True` entrena con menos
datos para probar que todo funciona; los resultados de acá están corridos con `False`.

---

## El problema

Clasificar imágenes de 28x28 en escala de grises en 10 tipos de prenda. La entrada es un
tensor `(28, 28, 1)`, la salida son 10 logits y la pérdida es
`SparseCategoricalCrossentropy(from_logits=True)`.

## El dataset

Fashion-MNIST, que se carga directo desde `tf.keras.datasets`.

| | |
|---|---|
| Imágenes | 60.000 de entrenamiento, 10.000 de prueba |
| Partición usada | 54.000 train / 6.000 validación / 10.000 test |
| Tamaño | 28 x 28, un solo canal |
| Clases | 10, con 6.000 imágenes cada una |
| Valores | enteros de 0 a 255 |

**Por qué sirve para convoluciones.** Las capas convolucionales dan por hecho que lo
importante está en zonas chicas de la imagen y que un detalle útil en una parte también
lo es en otra. En ropa las dos cosas se cumplen: lo que separa una bota de una sandalia
es la altura de la caña o los huecos entre tiras, no la relación entre dos píxeles
opuestos.

Para no quedarme solo con eso, en la sección 2.4 del notebook comparo cuánto se parecen
píxeles cercanos contra píxeles lejanos. Los vecinos se parecen mucho más, y la semejanza
cae rápido con la distancia, así que mirar solo vecindarios chicos no está dejando afuera
información importante.

**Por qué no MNIST.** Era lo obvio porque ya lo habíamos usado en clase, pero ahí una red
densa simple ya pasa del 97% y queda muy poco margen para que se note la mejora.
Fashion-MNIST tiene el mismo formato y es más difícil, sobre todo porque varias clases se
parecen entre sí.

**Preprocesamiento:** dividir entre 255, agregar la dimensión de canal y separar 10% para
validación. Nada más. No redimensiono (ya son todas iguales) ni aumento datos, para que
la única diferencia entre los modelos sea la arquitectura. El conjunto de prueba lo uso
una sola vez por modelo, al final.

---

## Los modelos

### Modelo base (sin convoluciones)

```
entrada (28,28,1) -> Flatten(784) -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, linear)
```

**109.386 parámetros.**

Lo importante acá es el `Flatten`: al aplanar, el píxel de arriba a la izquierda y el de
abajo a la derecha quedan como dos posiciones cualesquiera de un vector. La red pierde
toda noción de qué píxeles eran vecinos antes de la primera capa.

Al principio había hecho este modelo más chico, pero así la comparación no servía: si la
CNN gana con el doble de parámetros, no sé si ganó por la convolución o por ser más
grande. Le puse un tamaño parecido al de la CNN a propósito.

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

**105.866 parámetros.**

Algo que no esperaba al hacer las cuentas: las dos capas convolucionales juntas son 4.800
parámetros, menos del 5% del total. Casi todo el modelo es la capa densa del final, y aún
así esas 4.800 son las que hacen la diferencia.

### Por qué cada decisión

| Decisión | Elegí | Por qué |
|---|---|---|
| capas conv | 2 | con 2 capas y sus pooling cada neurona final ve una zona de 10x10 sobre 28x28, que alcanza para cubrir una parte de la prenda. Con 1 la ventana es muy chica y con 4 el mapa se achica demasiado rápido |
| kernel | 3x3 | los píxeles dejan de parecerse a los pocos píxeles de distancia, así que una ventana chica debería alcanzar. Además dos capas de 3x3 ven lo mismo que una de 5x5 con menos parámetros. Es lo que pruebo en el experimento |
| filtros | 16 → 32 | los duplico cuando el pooling reduce el mapa a la mitad, para no perder tanta información de golpe. Arranco en 16 porque la primera capa solo detecta bordes |
| stride | 1 en conv, 2 en pooling | con stride 1 la convolución no se saltea nada; achicar la imagen se lo dejo al pooling, que no tiene parámetros |
| padding | `same` | sin padding cada conv recorta el borde, donde a veces todavía hay prenda. Además hace que el tamaño de salida no dependa del kernel, que me sirve para el experimento |
| activación | ReLU | con tanto fondo negro muchas activaciones van a ser cero igual, y ReLU no satura para valores positivos |
| salida | `linear` | va con `from_logits=True`, que es la forma estable. El softmax lo aplico después |
| pooling | MaxPooling 2x2 | achica el cómputo, agranda la zona que ve cada neurona sin agregar parámetros y hace que un corrimiento chico no cambie el resultado. Elegí max en vez de average porque me interesa si el patrón aparece, y con tanto fondo negro el promedio lo diluiría |

### CNN compacta (control)

Los dos modelos de arriba comparten la misma capa densa de 100.000 parámetros, así que me
quedó la duda de cuánto de la mejora viene de la convolución y cuánto de esa capa. Armé
una tercera versión que reemplaza `Flatten + Dense(64)` por `GlobalAveragePooling2D`:

```
Conv(16) -> Pool -> Conv(32) -> Pool -> Conv(64) -> GlobalAvgPool -> Dense(10)
```

**23.946 parámetros**, casi 5 veces menos que el modelo base. La idea era: si igual se le
acerca, entonces la ventaja no es cuestión de tamaño.

**No funcionó.** Llegó a 0.8266 contra 0.8769 del modelo base, cinco puntos abajo. Pero
terminó con 0.8609 en entrenamiento y 0.8377 en validación, las dos todavía subiendo en la
última época y sin nada de sobreajuste, mientras las otras dos redes ya se habían aplanado.
O sea que quedó a medio entrenar: `GlobalAveragePooling2D` tira de golpe toda la
información de dónde estaba cada activación y la red tarda más en compensarlo.

Así que este control **no responde la pregunta que quería responder**, y lo dejo anotado
como tal en vez de forzar una conclusión. La comparación principal (modelo base contra CNN
con la misma cantidad de parámetros) no depende de él.

---

## Experimento: el tamaño del kernel

Probé 3x3 contra 5x5 contra 7x7, dejando fijo todo lo demás (2 capas conv, 16 y 32
filtros, stride 1, padding `same`, pooling, ReLU, la misma capa densa, Adam con lr = 1e-3,
batch 128, mismas épocas, misma partición).

Dos cosas del diseño que vale la pena aclarar:

**El padding `same` es a propósito.** Hace que el tamaño de salida no dependa del kernel,
así el vector que entra a la capa densa siempre mide 1568 y esa capa tiene los mismos
parámetros en las tres pruebas. Lo único que cambia son los parámetros de las
convoluciones. Si no fuera así estaría cambiando dos cosas a la vez.

**Corro cada configuración 3 veces.** La primera vez entrené una sola vez cada kernel y me
dio que 5x5 era el mejor; después volví a correr lo mismo y me dio distinto. Como los
pesos arrancan al azar, con una sola corrida no se puede distinguir una diferencia real de
la casualidad.

### Lo que cuesta cada kernel

| Kernel | Parámetros conv | Parámetros totales | Zona que ve cada neurona |
|---|---:|---:|---:|
| 3x3 | 4.800 | 105.866 | 10 x 10 |
| 5x5 | 13.248 | 114.314 | 16 x 16 |
| 7x7 | 25.920 | 126.986 | 22 x 22 |

### Resultados

Corrido con 15 épocas, batch 128, Adam con lr = 1e-3. Los números completos quedan en
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

### Análisis

**En exactitud los tres kernels empatan.** La diferencia entre el mejor y el peor es
0.0024, y las corridas del mismo kernel varían hasta 0.0061 entre sí solo por cómo
arrancaron los pesos. La diferencia es más chica que el ruido, así que no se puede afirmar
que un kernel sea mejor que otro.

Acá es donde sirvió correr tres semillas: con una sola habría visto que 3x3 quedó primero
y habría concluido que gana, cuando esa diferencia no significa nada.

**Donde sí hay diferencia clara es en el costo.** Pasar de 3x3 a 7x7 multiplica por 5,4
los parámetros de las convoluciones y duplica el tiempo, para el mismo resultado. Y la
brecha entre entrenamiento y validación crece de +0.0240 a +0.0308, o sea que los kernels
grandes se sobreajustan un poco más.

**Mi hipótesis quedó respaldada, pero de otra forma.** Yo esperaba que 3x3 alcanzara. No
es que 3x3 gane: es que agrandar el kernel no aporta nada mientras cuesta cinco veces más.
Cuando dos opciones dan el mismo resultado, se elige la barata.

Esto encaja con lo que había visto al medir la correlación entre píxeles: a 5 píxeles
todavía quedaba 0.41, así que era esperable que 5x5 no fuera peor. Lo que no le alcanza es
para ser mejor.

| Kernel | A favor | En contra |
|---|---|---|
| 3x3 | el mismo resultado con 5 veces menos parámetros y la mitad de tiempo | ve poco por capa, necesita apilar |
| 5x5 | ve más zona de una | 2,8 veces más parámetros para el mismo resultado |
| 7x7 | mucho contexto inmediato | 5,4 veces más parámetros, el doble de tiempo, algo más de sobreajuste, y el mismo resultado |

**Conclusión:** conviene 3x3, no porque dé mejor exactitud sino porque da la misma mucho
más barato. Si hiciera falta ver una zona más grande, lo sensato sería apilar capas chicas
antes que agrandar el kernel.

---

## Interpretación

### Una prueba extra: mover las imágenes

Se me ocurrió mientras escribía el análisis. Corrí las imágenes de prueba unos píxeles,
rellenando con negro, y evalué los dos modelos sin volver a entrenarlos.

| Corrimiento | Baseline | CNN | Ventaja de la CNN |
|---|---:|---:|---:|
| 0 px | 0.8769 | 0.9006 | +0.024 |
| 1 px | 0.7452 | 0.8357 | +0.091 |
| 2 px | 0.4273 | 0.6623 | **+0.235** |
| 3 px | 0.2265 | 0.3030 | +0.077 |
| 4 px | 0.1458 | 0.1733 | +0.028 |

El resultado tiene dos partes. **Con corrimientos chicos la CNN aguanta bastante mejor**: a
2 píxeles el denso se cayó a 0.43 y la CNN sigue en 0.66, 23 puntos de diferencia. **Pero a
3 y 4 píxeles se caen las dos**, hasta quedar por debajo del 20%, que con 10 clases es
apenas mejor que tirar una moneda.

La tolerancia viene de los dos `MaxPooling2D` de 2x2, que dan margen de unos pocos píxeles,
y justo hasta ahí llega la ventaja. Más allá no hay nada que la sostenga, porque la CNN
termina en un `Flatten` seguido de una capa densa, y esa capa sí depende de en qué posición
apareció cada activación.

Esto me aclaró algo que tenía confuso: la convolución no regala invarianza a la traslación.
Da que el mismo filtro detecte el mismo patrón en cualquier lado, que no es lo mismo. Si
después aplanás y conectás a una densa, la posición vuelve a importar. Los supuestos no
vienen solo del tipo de capa, vienen de cómo armás la red entera.

### ¿Por qué la CNN funcionó mejor?

La CNN llegó a 0.9006 contra 0.8769 del modelo denso: dos puntos y medio, con 3.520
parámetros menos. De los tres argumentos que quería usar, dos se sostienen y uno no:

1. **Con la misma cantidad de parámetros la CNN gana** (105.866 contra 109.386). Si solo
   importara el tamaño, deberían haber empatado. ✅
2. **La versión compacta no le llegó al modelo base.** Quedó a medio entrenar, así que no
   sirve como argumento ni a favor ni en contra. ❌
3. **La CNN aguanta mejor los corrimientos chicos**, con la salvedad de arriba. ✅

**Dónde gana exactamente.** Las mejoras están en pullover (+9,3 puntos), camiseta (+7,3) y
camisa (+2,9), justo las prendas de torso que comparten silueta y se distinguen por
textura. En las clases fáciles (pantalón, botín, zapatilla) mejora medio punto o menos
porque no había margen, y en abrigo queda dos puntos **peor**.

Eso es lo más interesante del resultado: la CNN no mejora parejo, mejora justo donde hacía
falta mirar el detalle fino, que era la predicción que había hecho mirando las imágenes
antes de entrenar.

Lo que creo que pasa por debajo: un filtro de 3x3 se aplica en las 784 posiciones de la
imagen, así que sus 9 pesos reciben información de todas esas posiciones en cada imagen. Un
peso de la capa densa solo ve un píxel por imagen.

**Aclaración honesta:** dos puntos y medio no es espectacular, por dos motivos. Las
imágenes son chicas y están centradas, así que el problema de la posición variable casi no
existe. Y el 95% de los parámetros de la CNN están en la capa densa final: en el fondo son
dos redes que comparten casi toda la estructura salvo el principio.

### ¿Qué supuestos trae la convolución?

Al elegir la arquitectura le metemos al modelo ideas nuestras sobre los datos antes de que
vea un ejemplo. En la convolución son cuatro:

| Supuesto | Cómo se mete en la red | Qué vi en mis pruebas |
|---|---|---|
| lo importante es local | cada neurona se conecta solo a una ventana de `k x k` | se cumple: la correlación cae a cero a los 10 px |
| lo que sirve acá sirve allá | el mismo filtro recorre toda la imagen | se cumple: gana con menos parámetros |
| mover la imagen no cambia qué es | sale de repetir los mismos pesos | **solo en parte**: aguanta 1-2 px, no 4 |
| lo complejo se arma con lo simple | apilar capas hace que cada una vea más zona | los filtros aprendidos son detectores de borde |

Lo que más me llamó la atención es que todos son **restricciones**: una capa convolucional
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

### ¿Cuándo no conviene?

Cuando esos supuestos no se cumplen:

- **Datos en tabla.** El orden de las columnas es arbitrario: no hay motivo para que edad
  e ingresos estén "al lado" ni para pasar el mismo filtro sobre columnas que miden cosas
  distintas. Mejor una red densa o árboles.
- **Cuando la posición exacta importa.** Si la tarea es "¿hay algo brillante arriba a la
  izquierda?", que la red responda igual sin importar dónde aparece el patrón es
  justamente lo que no quiero. Pasa en imágenes médicas donde la posición anatómica es
  parte del diagnóstico.
- **Cuando lo que importa está lejos.** En texto el sujeto y el verbo pueden estar
  separados por muchas palabras; una CNN necesitaría muchísimas capas para llegar. Para
  eso están los mecanismos de atención.
- **Cuando los datos no forman una grilla.** Grafos, redes sociales, nubes de puntos: no
  existe "el de al lado a la derecha".

La pregunta que me llevo, más útil que "si son imágenes usá CNN", es: **¿los datos tienen
vecinos que significan algo, y un patrón que sirve en un lugar sirve también en otro?** Si
las dos respuestas son sí, la convolución ayuda. Si alguna es no, esos supuestos se
vuelven una limitación.

---

## Filtros aprendidos (bonus)

La sección 7 del notebook muestra los pesos de la primera capa y los mapas de activación.
Los filtros no son ruido: se ven patrones con una parte clara y otra oscura en distintas
direcciones, que es la pinta de un detector de bordes, y nadie los programó. En los mapas
de `conv_1` todavía se reconoce la silueta de la prenda; en `conv_2` ya son más chicos y
más difíciles de interpretar.

---

## Despliegue en SageMaker

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

### Cómo se corrió: entrenamiento containerizado sin AWS

Me quedé sin créditos del curso, así que no pude usar ni las instancias de SageMaker ni su
Local Mode (que igual necesita credenciales de AWS para bajarse la imagen del contenedor
desde ECR).

Lo que hice fue reproducir el esquema con la imagen pública `tensorflow/tensorflow:2.13.0`
de Docker Hub, que no pide autenticación:

```bash
python entrenar_en_docker.py --epochs 15
```

SageMaker en modo script no hace nada especial: mete tu `train.py` en un contenedor, monta
los datos en `/opt/ml/input/data/{train,validation}`, define las variables `SM_MODEL_DIR`,
`SM_CHANNEL_TRAIN`, `SM_CHANNEL_VALIDATION` y `SM_OUTPUT_DATA_DIR`, y lo ejecuta. El script
[`entrenar_en_docker.py`](entrenar_en_docker.py) arma exactamente ese contenedor.

**El `sagemaker/train.py` no se modificó ni una línea**: es el mismo archivo que usaría un
training job real. El contenedor deja el `SavedModel` en `salida_docker/model/1`, igual que
SageMaker lo dejaría en `/opt/ml/model`.

Resultado de esa corrida: `val_accuracy` 0.9080, y cargando el `SavedModel` que produjo el
contenedor da **0.919** sobre 2.000 imágenes de prueba.

**Lo que esto no es:** un endpoint de SageMaker. Eso requiere una cuenta con créditos. Lo
que sí demuestra es que el script de entrenamiento funciona containerizado y que el modelo
exportado es válido y se puede servir.

El notebook [`sagemaker_train_deploy.ipynb`](sagemaker_train_deploy.ipynb) queda completo y
listo para correr en cuanto haya créditos: tiene la bandera `MODO_LOCAL` para alternar
entre Docker y las instancias de AWS.

> Si instalás el SDK, fijalo a `sagemaker>=2.190,<3`. La versión 3 eliminó `sagemaker.local`,
> `sagemaker.tensorflow`, `Session` y `get_execution_role`, así que el notebook falla en la
> primera celda.

El notebook tiene una bandera para pasar de uno a otro:

| | `MODO_LOCAL = True` | `MODO_LOCAL = False` |
|---|---|---|
| Dónde entrena | contenedor con Docker | instancia `ml.m5.xlarge` |
| Endpoint | contenedor local en un puerto | endpoint de SageMaker |
| Datos | carpeta local con `file://` | subidos a S3 |
| Costo | no gasta créditos | se cobra por hora |

Para pasarlo a la nube alcanza con cambiar esa variable: lo único que cambia por debajo es
el `instance_type` y de dónde se leen los datos.

**Para correrlo en local hace falta:** Docker Desktop abierto, `pip install
"sagemaker[local]"` y credenciales de AWS configuradas. Lo último me sorprendió: aunque
todo se ejecute en mi máquina, el SDK igual necesita autenticarse para bajarse la imagen
del contenedor la primera vez. Bajar la imagen no consume créditos de cómputo y después
queda en caché.

La primera corrida tarda bastante porque la imagen de TensorFlow pesa varios gigas.

### Detalles del despliegue

| Etapa | Detalle |
|---|---|
| Datos | un `.npz` por canal |
| Entrenamiento | contenedor TensorFlow 2.13, modo script |
| Código | `sagemaker/train.py`, con la misma arquitectura del notebook |
| Métricas | salen de los logs con `metric_definitions`; en la nube quedan en CloudWatch |
| Endpoint | TensorFlow Serving |

En modo nube usaría CPU y no GPU, porque con 106.000 parámetros e imágenes de 28x28 la GPU
no se aprovecha.

La decisión que me parece más interesante es dónde poner el softmax. El modelo devuelve
logits porque se entrenó con `from_logits=True`, que es la forma estable. Si lo hubiera
metido adentro del modelo tendría que haber cambiado la pérdida. Dejándolo en
`output_handler` me quedan las dos cosas: entrenamiento estable y una respuesta que se
entiende del otro lado.

**Ojo si se corre en la nube:** el endpoint se cobra por hora mientras exista. La última
celda del notebook lo apaga, conviene correrla siempre al terminar.

---

## Lo que quedó afuera

Lo primero son las dos cosas que salieron distinto de lo previsto y quedaron sin resolver:

- **Entrenar la CNN compacta hasta que converja.** Es lo que más me interesa, porque es el
  control que quedó sin responder. Terminó todavía subiendo.
- **Medir la robustez al corrimiento de la CNN compacta.** Como usa
  `GlobalAveragePooling2D` en vez de `Flatten`, debería aguantar mucho mejor los
  corrimientos grandes. Es una predicción concreta que sale del análisis y que no llegué a
  probar.

Y después:

- Probé un solo dataset y de imágenes chicas. No sé si lo del kernel se mantendría con
  imágenes más grandes; tengo entendido que ahí sí se usan kernels más grandes en las
  primeras capas, pero no llegué a probarlo.
- Tres corridas por configuración sirven para ver si la diferencia supera al ruido, pero
  no es una prueba estadística en serio.
- No usé aumento de datos, dropout ni batch normalization a propósito, para aislar el
  efecto de la arquitectura. Me quedó la duda de cuánto mejoraría el modelo denso con algo
  de regularización, porque es el que más se sobreajusta.
- En los mapas de la segunda capa no pude interpretar qué detecta cada filtro.

**Lo que probaría después:** entrenar el modelo denso con las imágenes corridas al azar,
para ver si aprende solo a aguantar los corrimientos o si de verdad hace falta que la
arquitectura se lo imponga.
