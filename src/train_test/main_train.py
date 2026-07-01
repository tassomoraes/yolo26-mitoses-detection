import os

from sympy import false
from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer

from constants import dataset_yaml_path, dataset_SEG_yaml_path, results_root, MODEL
from functions import load_model_train


import torch
from ultralytics import YOLO

BASE_DIR = "/media/ampliar/e970332c-7024-4ab5-bf4c-ef01d5ba9509/workspace/mitoses-detection/YOLO-Colonoscopy"

# from ultralytics import settings

# settings.update({'tensorboard': True})

if __name__ == '__main__':

    # If needed for frozen executables, uncomment the next line:
    # from multiprocessing import freeze_support; freeze_support()

    model, model_name, model_path = load_model_train(MODEL)
    print(model.names)

    results_root_ = f"{results_root}/runs_train/{MODEL}/"
    results_root_ = os.path.abspath(results_root_)
    os.makedirs(results_root_, exist_ok=True)

    
    # Train the model with enhanced parameters and augmentation
    results = model.train(
        data=dataset_yaml_path if MODEL != "yoloe" else dataset_SEG_yaml_path,  # dataset YAML config
        epochs=30,              # number of epochs
        imgsz=640,            # training image size
        batch=16,               # adjust according to your GPU memory
        optimizer="AdamW",      # try different optimizers
        lr0=0.0001,              # initial learning rate
        lrf=0.1,               # final learning rate as a fraction of initial lr
        
        # regularização — principais ajustes
        dropout=0.1,          # era 0.0 — ativa dropout nas camadas finais
        weight_decay=0.001,   # era 0.0005 — aumenta penalização L2

        momentum=0.9,         # SGD momentum
        warmup_epochs=3.0,        # warmup epochs
        warmup_bias_lr=0.0001,
        plots=True,
        cos_lr=True,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        trainer=None if MODEL != "yoloe" else YOLOEPESegTrainer,

        # recall vs precision — threshold de confiança no val
        conf=0.1,             # padrão é 0.25 — abaixa para não filtrar detecções válidas
        amp=False,


        # Early stopping patience (if needed)
        patience=15,            # early stopping patience (epochs)

        # Save best model during training
        save=True,              # save best model
        save_period=5,         # save checkpoint every x epochs
        project=results_root_,
        name=model_name,        # experiment name
    )
    
    # Resume training from the last checkpoint
    '''
    tain_number = '18'
    ckpt_path = f"{BASE_DIR}/results_data_MIDOGpp/runs_train/yolo26/yolo26s-{tain_number}/weights/epoch10.pt"

    
    ckpt = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    print("Último epoch salvo :", ckpt.get('epoch'))
    print("Melhor mAP50       :", ckpt.get('best_fitness'))


    model = YOLO(
        f"{BASE_DIR}/results_data_MIDOGpp/runs_train/yolo26/yolo26s-{tain_number}/weights/epoch10.pt"
    )

    results = model.train(
        resume=True,
        epochs=30,    # total acumulado, não adicional
        patience=0,     # desativa early stopping
    )
    ''' 
    print(model.names)
    print("Training completed.")

    # Save the current state of the model to a file.
    model.save(model_path)

    # evaluation_results = model.val(
    #     data=dataset_yaml_path if MODEL != "yoloe" else dataset_SEG_yaml_path,
    #     imgsz=640,
    #     project="runs",
    #     name=model_name,
    # )
    #
    # print(evaluation_results)
