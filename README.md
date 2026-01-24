Progetto d'Esame "Laboratorio IA" - Object Detection

Descrizione

Questo progetto confronta tre modelli per l'object detection sul dataset VisDrone2019-DET:
- YOLOv11 (Ultralytics)
- RetinaNet
- Faster R-CNN

L'obiettivo e' valutare le prestazioni di detection e documentare le scelte di training (iperparametri, scheduler, augmentations, ecc.).

Struttura del Progetto

- config/
  - config.json: configurazione principale (dataset, modelli, training)
  - config_schema.json: schema di validazione
- data/
  - data_loader.py: loader dei dataset per detection
  - detection_transforms.py: transforms per immagini e bounding box
  - visdrone_dataset.py: parser del formato VisDrone
  - visdrone_to_yolo.py: conversione VisDrone -> YOLO (opzionale)
- models/
  - detectors.py: factory per RetinaNet e Faster R-CNN
- codes/
  - train_detection.py: training loop per modelli TorchVision
  - train_yolo.py: training loop per YOLOv11
  - evaluate_detection.py: validazione e metriche
  - metrics.py: mAP/metriche fallback
- utils/
  - early_stopping.py: early stopping e checkpoint
  - checkpoints.py: utilita' checkpoint
  - class_names.py: classi VisDrone

Requisiti

Usa il file `environmental.yaml` per creare l'ambiente conda:

```bash
conda env create -f environmental.yaml
```

Dataset

Scarica VisDrone2019-DET e organizza le cartelle:

```
data/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
```

Per YOLOv11 e' disponibile una conversione automatica verso il formato YOLO in:
`data/VisDrone2019-DET-YOLO/`. Il percorso e' configurabile in `config/config.json`.

Nota: `data.num_classes` indica il numero di classi oggetto (10 per VisDrone).
Faster R-CNN aggiunge automaticamente la classe background.
Per RetinaNet i label vengono convertiti a 0-based durante il training.

Esecuzione Training

Faster R-CNN:
```bash
python main.py --model faster_rcnn
```

RetinaNet:
```bash
python main.py --model retinanet
```

YOLOv11:
```bash
python main.py --model yolov11
```

Esecuzione esperimenti predefiniti:
```bash
python main.py --model faster_rcnn
```

Oppure un singolo esperimento:
```bash
python main.py --model faster_rcnn --resume
```

Ripresa da checkpoint:
```bash
python main.py --model faster_rcnn --resume
```

Nota: per fermare l'addestramento su mAP anziche' loss, imposta
`training.early_stop_metric` su `map_50` e `training.early_stop_mode` su `max`.

Inferenza su immagini singole

```bash
python image_test.py --model faster_rcnn --image path/to/image.jpg
```

Output training (TorchVision):
- `runs/<model>/<timestamp>_<experiment-name>/results.csv` con loss, mAP, precision/recall/F1 e learning rate per epoca.
- `runs/<model>/<timestamp>_<experiment-name>/predictions/epoch_<N>/` con immagini annotate (GT in verde, pred in rosso).
- `runs/<model>/<timestamp>_<experiment-name>/tb/` con i log TensorBoard.
- `runs/<model>/latest_run.txt` contiene il path dell'ultimo run (usato per `--resume`).
