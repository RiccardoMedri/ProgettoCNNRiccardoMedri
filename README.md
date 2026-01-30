# Progetto d'Esame "Laboratorio IA" - Object Detection (VisDrone2019-DET)

## Panoramica
Questo progetto confronta tre modelli di object detection sul dataset VisDrone2019-DET:
- YOLOv11 (Ultralytics)
- RetinaNet (TorchVision)
- Faster R-CNN (TorchVision)

L'obiettivo e' misurare prestazioni di detection (mAP, precision/recall, tempi) e documentare le scelte di training (iperparametri, scheduler, augmentations).

## Struttura
- `config/`: configurazione e schema
- `data/`: loader, trasformazioni e parser VisDrone
- `models/`: factory modelli
- `codes/`: training e valutazione
- `utils/`: utility (checkpoint, classi, visualizzazioni)
- `runs/`: output training/valutazione
- `Test/`: input per inference a dataset (images/annotations/predictions)

## Requisiti e setup
Usa il file `environment.yaml` per creare l'ambiente conda:
```bash
conda env create -f environment.yaml
```

## Dataset VisDrone2019-DET
Struttura attesa:
```
data/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
```

Per YOLOv11 e' disponibile la conversione automatica in formato YOLO sotto:
`data/VisDrone2019-DET-YOLO/`. Il percorso e' configurabile in `config/config.json`.

## Avvio training (main.py)
Script principale per l'addestramento.

### Sintassi
```bash
python main.py --model <yolov11|retinanet|faster_rcnn> [--config PATH] [--resume]
```

### Opzioni disponibili
- `--model`: modello da allenare. Valori: `yolov11`, `retinanet`, `faster_rcnn`.
- `--config`: path del file di configurazione (default: `config/config.json`).
- `--resume`: riprende da checkpoint. Richiede `training.resume_checkpoint_path` valorizzato nel config.

### Esempi
```bash
python main.py --model faster_rcnn
python main.py --model retinanet
python main.py --model yolov11
python main.py --model faster_rcnn --resume
```

## Inferenza e valutazione (image_test.py)
Script per inferenza su singola immagine o dataset, con salvataggio overlay e metriche.

### Sintassi
```bash
python image_test.py --model <yolov11|retinanet|faster_rcnn> [--config PATH] [--score-threshold N]
                    [--skip-metrics] [--skip-predict]
                    (--image PATH | --dataset PATH)
```

### Opzioni disponibili
- `--model`: modello da usare in inference. Valori: `yolov11`, `retinanet`, `faster_rcnn`.
- `--config`: path del file di configurazione (default: `config/config.json`).
- `--score-threshold`: soglia di confidenza per filtrare predizioni (se omesso usa config).
- `--skip-metrics`: salta il calcolo metriche.
- `--skip-predict`: salta la generazione delle immagini con overlay.
- `--image`: path di una singola immagine (richiede anche le annotation VisDrone).
- `--dataset`: path a un dataset in `Test/<nome>` con `images/`, `annotations/`, `predictions/`.

### Esempi
```bash
python image_test.py --model faster_rcnn --image data/VisDrone2019-DET-val/images/0000001.jpg
python image_test.py --model yolov11 --dataset Test/val_sample
python image_test.py --model retinanet --dataset Test/val_sample --skip-metrics
python image_test.py --model retinanet --image path/to/image.jpg --score-threshold 0.3
```

## Output principale
- Training TorchVision:
  - `runs/<model>/<timestamp>_<model>/results.csv`: loss, mAP, precision/recall/F1 e learning rate per epoca.
  - `runs/<model>/<timestamp>_<model>/predictions/epoch_<N>/`: immagini annotate (GT verde, pred rosso).
  - `runs/<model>/<timestamp>_<model>/tb/`: log TensorBoard.
- Inference con `image_test.py`:
  - su immagine singola: file `*_gt_pred_<model>.png` accanto all'immagine.
  - su dataset: `Test/<nome>/predictions/predict_<model>_<timestamp>/` con overlay e `metrics.json`.

## Riferimento completo `config/config.json`
Di seguito ogni valore di configurazione con significato e opzioni ammesse.

### experiment
- `experiment.seed` (int): seed per la riproducibilita'. Opzioni: qualsiasi intero.
- `experiment.device` (string): device preferito. Opzioni: `cuda` o `cpu` (se `cuda` non e' disponibile usa `cpu`).

### data
- `data.batch_size` (int): batch size per training/val. Opzioni: intero > 0.
- `data.num_workers` (int): worker per DataLoader. Opzioni: intero >= 0.
- `data.image_size` (int): dimensione base di resize. Opzioni: intero > 0.
- `data.num_classes` (int): numero classi oggetto (VisDrone = 10). Opzioni: intero > 0.
- `data.class_map` (dict): mappa classi VisDrone -> classi target. Opzioni: mappa `{"1":1, ...}`; usata nel parser annotazioni.
- `data.valid_categories` (list[int]): categorie accettate. Opzioni: lista di ID classe.
- `data.normalize.mean` (list[float]): media canali RGB per normalizzazione. Opzioni: 3 float.
- `data.normalize.std` (list[float]): deviazione standard canali RGB. Opzioni: 3 float.
- `data.augmentation.horizontal_flip` (bool): abilita flip orizzontale. Opzioni: `true|false`.
- `data.augmentation.flip_prob` (float): probabilita' di flip. Opzioni: 0.0 - 1.0.
- `data.augmentation.color_jitter` (bool): abilita jitter colore. Opzioni: `true|false`.
- `data.augmentation.brightness` (float): intensita' jitter brightness. Opzioni: >= 0.
- `data.augmentation.contrast` (float): intensita' jitter contrast. Opzioni: >= 0.
- `data.augmentation.saturation` (float): intensita' jitter saturation. Opzioni: >= 0.
- `data.augmentation.hue` (float): intensita' jitter hue. Opzioni: 0.0 - 0.5 circa (TorchVision).
- `data.train.images_dir` (string): path immagini training. Opzioni: path valido.
- `data.train.annotations_dir` (string): path annotazioni training. Opzioni: path valido.
- `data.val.images_dir` (string): path immagini validation. Opzioni: path valido.
- `data.val.annotations_dir` (string): path annotazioni validation. Opzioni: path valido.
- `data.yolo.dataset_root` (string): root dataset YOLO. Opzioni: path valido.
- `data.yolo.auto_convert` (bool): se `true` converte automaticamente VisDrone -> YOLO. Opzioni: `true|false`.
- `data.yolo.yaml_path` (string): path file YAML dataset YOLO. Opzioni: path valido.

### training
- `training.epochs` (int): numero epoche. Opzioni: intero > 0.
- `training.patience` (int): patience per early stopping. Opzioni: intero >= 0.
- `training.delta` (float): miglioramento minimo per early stopping. Opzioni: float >= 0.
- `training.early_stop_mode` (string): direzione metrica. Opzioni: `max` o `min`.
- `training.early_stop_metric` (string): metrica monitorata. Opzioni tipiche: `map_50_95`, `map_50`, `loss`.
- `training.use_amp` (bool): abilita mixed precision. Opzioni: `true|false`.
- `training.gradient_clip_norm` (float): norma max per gradient clipping. Opzioni: float >= 0.
- `training.gradient_accumulation_steps` (int): step di accumulo gradiente. Opzioni: intero >= 1.
- `training.log_interval` (int): frequenza log (batch). Opzioni: intero >= 1.
- `training.save_predictions_every` (int): ogni quante epoche salva predizioni. Opzioni: intero >= 1.
- `training.prediction_samples` (int): numero immagini campione per predizioni. Opzioni: intero >= 1.
- `training.prediction_score_threshold` (float): soglia score per predizioni salvate. Opzioni: 0.0 - 1.0.
- `training.runs_dir` (string): cartella output run. Opzioni: path valido.
- `training.resume_checkpoint_path` (string|null): path checkpoint da cui riprendere. Opzioni: path valido o `null`.
- `training.warmup_epochs` (int): numero epoche warmup LR. Opzioni: intero >= 0.
- `training.warmup_start_factor` (float): fattore iniziale warmup LR. Opzioni: float > 0.
- `training.optimizer.lr` (float): learning rate iniziale. Opzioni: float > 0.
- `training.optimizer.min_lr` (float): learning rate minimo per CosineAnnealing. Opzioni: float >= 0.
- `training.optimizer.weight_decay` (float): weight decay. Opzioni: float >= 0.

### evaluation
- `evaluation.iou_threshold` (float): IOU per mAP/metriche. Opzioni: 0.0 - 1.0.
- `evaluation.score_threshold` (float): soglia score per valutazione. Opzioni: 0.0 - 1.0.
- `evaluation.track_inference_time` (bool): misura tempo inferenza in validazione. Opzioni: `true|false`.

### prediction
- `prediction.score_threshold` (float): soglia score in inference se non passata da CLI. Opzioni: 0.0 - 1.0.
- `prediction.eval_use_score_threshold` (bool): usa la soglia anche durante valutazione YOLO. Opzioni: `true|false`.
- `prediction.model_weights.yolov11` (string): pesi per inferenza YOLO. Opzioni: path valido o nome modello Ultralytics.
- `prediction.model_weights.retinanet` (string): pesi per inferenza RetinaNet. Opzioni: path valido, `coco`, `default`, `none`.
- `prediction.model_weights.faster_rcnn` (string): pesi per inferenza Faster R-CNN. Opzioni: path valido, `coco`, `default`, `none`.

### models
- `models.yolov11.type` (string): tipo modello. Opzioni: `yolov11`.
- `models.yolov11.weights` (string): pesi YOLO. Opzioni: path valido o nome modello Ultralytics (es. `yolov11s.pt`).
- `models.retinanet.type` (string): tipo modello. Opzioni: `retinanet`.
- `models.retinanet.weights` (string): pesi RetinaNet. Opzioni: path valido, `coco`, `default`, `none`.
- `models.retinanet.trainable_backbone_layers` (int): layer backbone allenabili. Opzioni: 0-5.
- `models.retinanet.transform.min_size` (int|list[int]): min size resize. Opzioni: int o lista di int.
- `models.retinanet.transform.max_size` (int): max size resize. Opzioni: int.
- `models.faster_rcnn.type` (string): tipo modello. Opzioni: `faster_rcnn`.
- `models.faster_rcnn.weights` (string): pesi Faster R-CNN. Opzioni: path valido, `coco`, `default`, `none`.
- `models.faster_rcnn.trainable_backbone_layers` (int): layer backbone allenabili. Opzioni: 0-5.
- `models.faster_rcnn.transform.min_size` (int|list[int]): min size resize. Opzioni: int o lista di int.
- `models.faster_rcnn.transform.max_size` (int): max size resize. Opzioni: int.
