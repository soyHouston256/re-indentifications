# Reidentificación no supervisada: MLCOR + CEM

Implementación independiente del método de Zhang, Cheng, Du y Yang, **Confidence-guided outlier refinement and collaborative embedding for unsupervised person re-identification**, Scientific Reports (2026), DOI: [10.1038/s41598-026-56072-w](https://doi.org/10.1038/s41598-026-56072-w).

Este repositorio incluye implementación, herramientas de reproducción, configuración del experimento completo y métricas agregadas. Dataset, checkpoints, imágenes, presentaciones y logs se mantienen fuera. Los resultados corresponden a la partición académica HDF5 suministrada para el Laboratorio 3; no son una reproducción exacta del benchmark publicado.

## Experimento completado

Se entrenaron **tres variantes, 50 épocas × 200 actualizaciones cada una**, semilla 42, P=4/K=4: batch de entrenamiento de 16 imágenes, 10 000 actualizaciones por modelo y 30 000 en total. Cada época extrae características de las 11 043 imágenes de entrenamiento, recalcula pseudoclases y realiza 200 actualizaciones. El batch de extracción/evaluación es 32.

La ejecución local usó PyTorch con **Metal/MPS en una Mac**, cuatro hilos CPU y duró **5 h 37 min 29 s**, del 20/10/2026 a las 22:14:22 al 21/10/2026 a las 03:51:51 (Lima). No requirió Kipu. Estos tiempos son observados, no una estimación para otros equipos.

Testing: métricas en porcentaje; caída en puntos porcentuales. Las 3368 consultas fueron válidas en cada condición.

| Variante | MLCOR | CEM | Época elegida | mAP original | Rank-1 original | mAP baja resolución | Rank-1 baja resolución | Caída mAP |
|---|---|---|---:|---:|---:|---:|---:|---:|
| full | Sí | Sí | 47 | 59.77 | 79.63 | 30.18 | 45.96 | 29.60 |
| baseline | No | No | 50 | **62.00** | **81.53** | **35.22** | **51.90** | 26.78 |
| no_mlcor | No | Sí | 46 | 59.46 | 79.45 | 33.83 | 50.56 | 25.63 |

El baseline superó al modelo completo en 2.23 pp de mAP original y 5.04 pp de mAP de baja resolución. La menor caída de `no_mlcor` no implica el mejor desempeño absoluto. Una sola semilla no permite establecer significancia estadística ni atribuir causalmente las diferencias a un componente.

Los valores completos de mAP, Rank-1/5/10, validación, tiempos y épocas seleccionadas están en [experiments/night_results.json](experiments/night_results.json). Son un resumen de la ejecución final, no métricas generadas al instalar el proyecto. El piloto previo de 1 época × 20 pasos solo comprobaba el pipeline; no debe mezclarse con esta tabla.

## Instalación y herramientas

Python 3.9 fue usado en el entorno local. Desde la raíz de este repositorio (la carpeta `code/` del laboratorio):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m unittest discover -s tests -v
```

[requirements-tested.txt](requirements-tested.txt) registra las versiones observadas: PyTorch 2.8.0, torchvision 0.23.0, NumPy 1.26.2, SciPy 1.11.4, h5py 3.14.0, scikit-learn 1.6.1 y Pillow 9.5.0. `threadpoolctl` controla los hilos BLAS durante evaluación y figura como dependencia explícita en ambos archivos de requisitos (versión 3.7.0 verificada al preparar esta entrega). `requirements.txt` ofrece rangos de compatibilidad, no un bloqueo del entorno completo. La disponibilidad de ruedas depende de Python y plataforma.

PyTorch/torchvision implementan el encoder y la optimización; h5py lee HDF5; NumPy/SciPy y DBSCAN de scikit-learn realizan el agrupamiento; Pillow prepara imágenes. La primera ejecución descarga los pesos **ResNet50 IMAGENET1K_V1** de torchvision si no están en caché. El entrenamiento no parte de pesos de reidentificación ni del código de los autores.

## Datos utilizados y procedencia

Se utilizaron los tres HDF5 entregados para el laboratorio, basados en Market-1501. Deben obtenerse por el canal académico original y conservar estos nombres fuera del repositorio:

```text
carpeta_padre/
├── code/                 # este repositorio
├── datos/
│   ├── training.h5
│   ├── validation.h5
│   └── testing.h5
└── resultados/           # se genera al ejecutar
```

| Archivo | Entrenamiento / galería | Consultas originales | Consultas de baja resolución |
|---|---:|---:|---:|
| training.h5 | 11 043 imágenes sin PID | — | — |
| validation.h5 | 1793 imágenes de galería | 100 | 100 |
| testing.h5 | 19 732 imágenes de galería | 3368 | 3368 |

Las imágenes son `uint8` CHW: originales 3×128×64, consultas degradadas 3×32×16. El modelo recibe ambas redimensionadas a 256×128; aumentar tamaño no recupera el detalle perdido. Entrenamiento contiene `images`, `camera_ids`, `image_ids`; **no contiene `person_ids`**. No se infieren identidades desde los nombres. Validación/testing contienen grupos `gallery` y `query`, con `images`, `person_ids`, `camera_ids`, `image_ids`; `query` añade `images_low_resolution`.

[experiments/dataset_manifest.json](experiments/dataset_manifest.json) registra el esquema exacto, tamaños y SHA-256 de los archivos utilizados. Permite identificar la partición sin publicar datos personales ni imágenes. Descargar otra distribución de Market-1501 no garantiza esta división HDF5. La referencia del dataset es Zheng et al., *Scalable Person Re-identification: A Benchmark*, ICCV 2015; el artículo del método está enlazado arriba.

## Reproducir el experimento completo

La configuración versionada [experiments/night.json](experiments/night.json) fija el presupuesto final. Usa un prefijo nuevo `reproduction` para separar nuevas ejecuciones del registro histórico `night`.

```bash
# Revisar esquema y comandos antes de entrenar
python -m reid.run inspect --data ../datos --output ../resultados/audit
python run_experiments.py --config experiments/night.json \
  --data ../datos --output ../resultados/reproduction --device mps --dry-run

# Entrenar las tres variantes, evaluar testing y generar informe
python run_experiments.py --config experiments/night.json \
  --data ../datos --output ../resultados/reproduction --device mps
```

`--dry-run` solo imprime los siete comandos. El ejecutor entrena secuencialmente `full`, `baseline`, `no_mlcor`; termina toda selección con validación antes de abrir testing. Los argumentos explícitos prevalecen sobre el JSON. `--device auto` selecciona MPS, CUDA o CPU; el tiempo y los resultados pueden variar entre dispositivos. No hay garantía de igualdad bit a bit con la misma semilla.

La selección utiliza la mayor media de mAP original y de baja resolución en validación (100 consultas por condición). Ambas condiciones de testing usan ese mismo `best.pt`. No se ajustan hiperparámetros con testing. La métrica usa el embedding global normalizado y similitud coseno, sin reranking; excluye PID=-1 y coincidencias de misma persona/misma cámara y omite consultas sin positivos válidos.

```bash
# Piloto breve explícito (no sustituye el experimento final)
python run_experiments.py --data ../datos --output ../resultados/pilot \
  --epochs 1 --steps 20 --prefix pilot

# Repetir solo la evaluación de un checkpoint existente
python -m reid.run evaluate --split testing --data ../datos \
  --checkpoint ../resultados/reproduction/reproduction_full/best.pt \
  --output ../resultados/reproduction/reproduction_full --device mps
```

Sin argumentos, el ejecutor carga `experiments/night.json`: 50×200 y P=4/K=4. El entrenamiento individual utiliza el mismo presupuesto por defecto. `--checkpoint` al entrenar carga únicamente el modelo como inicio; **no reanuda** optimizador, scheduler, época ni estado aleatorio. Usar directorios nuevos: las herramientas pueden sobrescribir artefactos de una ejecución anterior.

## Método e hiperparámetros

- ResNet50 ImageNet V1; GeM entrenable con p inicial 3, limitado a [1,6]; ramas global, mitad superior e inferior, tres BN necks y embeddings normalizados de 2048 dimensiones.
- Aumento: flip horizontal, padding/crop, normalización ImageNet y RandomErasing p=0.5.
- Distancia Jaccard de vecinos recíprocos k1=30/k2=6, fusión global/superior/inferior 0.70/0.15/0.15; DBSCAN eps=0.6, min_samples=4.
- MLCOR: cinco vecinos, sigma=0.5, umbrales 0.4/0.6/0.8, consenso 0.6 y hasta 40 recuperaciones por época; se prioriza alta confianza. Las etiquetas permanecen fijas durante cada ronda de refinamiento.
- CEM combina GEL (temperatura 0.05) y C-DTL (beta=0.2), con memorias de instancias/centroides y momentum 0.1. Objetivo: LBASE + C-DTL + 0.6 GEL; LBASE pondera ramas 0.8/0.2/0.2.
- Adam lr=3.5e-4, weight_decay=5e-4; StepLR cada 20 épocas, gamma=0.1. Semilla 42, cuatro hilos CPU, extracción/evaluación batch 32.

**Adaptaciones respecto del paper:** LBASE utiliza centroides normalizados como prototipos a temperatura 0.05 en vez de un clasificador entrenable. La ecuación 20 ya expresa log-probabilidad: C-DTL usa ese valor directamente y evita aplicar un segundo logaritmo inválido en la ecuación 24. La distancia positiva usa stop-gradient. No se añade un teacher independiente. Se utiliza una partición académica de 11 043 imágenes de entrenamiento y cómputo MPS, frente a las condiciones originales del artículo. Estas diferencias limitan la comparación directa con su tabla de resultados.

## Organización y salidas

| Archivo | Responsabilidad |
|---|---|
| `run_experiments.py` | Ejecutar las tres variantes, evaluar e informar; JSON y simulación de comandos |
| `reid/data.py` | HDF5 y transformaciones |
| `reid/model.py` | Encoder ResNet50/GeM y tres ramas |
| `reid/clustering.py` | Jaccard, DBSCAN y refinamiento MLCOR |
| `reid/losses.py` | Memorias, LBASE, GEL y C-DTL |
| `reid/run.py` | Muestreo PK, entrenamiento, selección y evaluación |
| `reid/evaluate.py` | Recuperación y protocolo de métricas |
| `reid/report.py` | Comparación y cinco ejemplos correctos/cinco incorrectos |
| `reid/paths.py` | Rutas externas predeterminadas |
| `tests/test_core.py` | Protocolo, refinamiento y pérdidas/gradientes |
| `experiments/` | Configuración, esquema/huellas del dataset y métricas agregadas |

Por variante se generan `config.json`, `history.json`, `validation_metrics.json`, `best.pt`, `last.pt`, etiquetas/estadísticas de clustering por época, métricas y ejemplos de testing. El informe produce `comparison.json`, `cases.json`, `RESULTADOS.md` y recursos visuales. Todo se escribe fuera del repositorio. Las configuraciones de evaluación pueden mostrar valores de entrenamiento por defecto: la variante realmente entrenada se identifica en `config.json` y en los argumentos guardados dentro del checkpoint.

Los `.gitignore` excluyen HDF5, pesos, arrays, entornos y documentos binarios. El informe y la PPT del laboratorio se conservan en la entrega académica, separados de esta implementación.
