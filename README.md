# MLCOR + CEM para reidentificación no supervisada

Implementación propia del método de Zhang et al., Scientific Reports (2026), DOI: 10.1038/s41598-026-56072-w.

## Instalación

Python 3.9 o superior. Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements-tested.txt` registra las versiones utilizadas en el piloto local.

## Datos externos

Preparar `training.h5`, `validation.h5` y `testing.h5` fuera de este repositorio. Por defecto, se buscan en `../datos/` y los resultados se escriben en `../resultados/`. Las rutas se pueden cambiar mediante argumentos. No se incluyen datos, pesos, resultados ni presentaciones.

## Ejecución

```bash
python -m reid.run inspect --data ../datos --output ../resultados/audit
python -m unittest discover -s tests -v

# Entrenamiento individual; solo validación elige el checkpoint
python -m reid.run train --variant full --data ../datos --output ../resultados/full

# Evaluación final con la configuración ya fijada
python -m reid.run evaluate --split testing --data ../datos \
  --checkpoint ../resultados/full/best.pt --output ../resultados/full

# Piloto secuencial: completo, baseline y ablación sin MLCOR
python run_experiments.py --data ../datos --output ../resultados \
  --epochs 1 --steps 20 --prefix pilot

# Presupuesto prolongado, separado del piloto
python run_experiments.py --data ../datos --output ../resultados \
  --epochs 50 --steps 200 --prefix extended
```

`--device auto` elige MPS, CUDA o CPU. Se puede indicar `--device mps`, `cuda` o `cpu`. El ejecutor comparativo mantiene P=4/K=4. El entrenamiento individual usa por defecto P=8/K=4; ambos valores se pueden configurar con `--p` y `--k`.

## Módulos

- `reid/data.py`: lectura HDF5 y transformaciones.
- `reid/model.py`: ResNet50, GeM y ramas global/locales.
- `reid/clustering.py`: Jaccard, DBSCAN y MLCOR.
- `reid/losses.py`: memorias, LBASE, GEL y C-DTL.
- `reid/run.py`: entrenamiento y selección con validación.
- `reid/evaluate.py`: recuperación y métricas Market-1501.
- `reid/report.py`: exportación de resultados y ejemplos a la carpeta externa.
- `reid/paths.py`: ubicaciones externas predeterminadas.
- `tests/`: pruebas del protocolo y de los componentes principales.

## Decisiones de implementación

Se conserva el mecanismo central MLCOR/CEM. LBASE usa centroides normalizados como prototipos y temperatura 0.05. La ecuación 20 ya define una log-probabilidad: en la ecuación 24 se utiliza directamente para evitar un segundo logaritmo inválido. La distancia positiva se trata con stop-gradient, siguiendo la separación exclusiva de negativos descrita por los autores. Jaccard usa k1=30/k2=6, DBSCAN min_samples=4 y GeM p inicial=3.

El entrenamiento no consume identidades reales. Las métricas usan embeddings L2 y similitud coseno, con exclusiones de misma identidad/misma cámara y PID=-1. Ambas resoluciones se evalúan con el mismo checkpoint. El piloto de una ronda y veinte pasos verifica el pipeline y no reproduce el presupuesto original del artículo.
