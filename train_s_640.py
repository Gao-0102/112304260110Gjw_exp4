import os

def main():
    os.chdir(r"d:\Trae CN Project\Trafficsign")

    os.environ['HOME'] = r"d:\Trae CN Project\Trafficsign"
    os.environ['USERPROFILE'] = r"d:\Trae CN Project\Trafficsign"
    os.environ['ULTRALYTICS_CONFIG_DIR'] = r"d:\Trae CN Project\Trafficsign\.ultralytics"
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'

    from ultralytics import YOLO

    model = YOLO('yolov8s.pt')

    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        device=0,
        batch=2,
        workers=0,
        patience=30,
        save=True,
        plots=False,
        verbose=True,
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.0,
        close_mosaic=10,
        project='runs',
        name='detect/train_s_640'
    )

    print("Training completed!")
    print(f"Best model saved at: runs/detect/train_s_640/weights/best.pt")

if __name__ == '__main__':
    main()