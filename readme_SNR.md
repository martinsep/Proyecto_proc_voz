# Voice Activity Detector (VAD)

Este proyecto proporciona un Voice Activity Detector (VAD) diseñado para procesar audios y calcular métricas comparativas entre los audios limpios y los audios procesados. A continuación, se describen los pasos para instalar el VAD y utilizar el script de métricas.

## Instalación

Para instalar el Voice Activity Detector, ejecuta el siguiente comando en tu terminal:

```bash
python install_vad.py
```


Este script descargará y configurará el VAD en tu entorno.

## Uso de metricas_audios.py

El script `metricas_audios.py` se utiliza para calcular métricas comparativas entre los audios limpios y los audios procesados.

### Parámetros

- `--referencia`: Ruta hacia la carpeta con los audios limpios.
- `--folder`: Ruta hacia la carpeta con los audios procesados.
- `--result_csv`: Nombre del archivo de salida que contendrá las métricas calculadas.
- `--vad` (opcional): Versión del VAD que se utilizará. Por defecto, se utiliza "VadOct27".

### Ejemplo de Uso

```bash
python metricas_audios.py --referencia /ruta/a/audios_limpios --folder /ruta/a/audios_procesados --result_csv resultado_metricas --vad VadOct27
```
Este comando calculará las métricas comparativas utilizando los audios limpios de la carpeta /ruta/a/audios_limpios, los audios procesados de la carpeta /ruta/a/audios_procesados, y guardará los resultados en el archivo CSV especificado como resultado_metricas.csv.