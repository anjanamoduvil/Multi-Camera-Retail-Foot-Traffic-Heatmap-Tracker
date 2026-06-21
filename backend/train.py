import argparse
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_yaml, model_size='n', epochs=100, imgsz=640, batch_size=16):
    """
    Fine-tune a YOLOv8 model on a custom dataset.
    
    Args:
        data_yaml: Path to the dataset.yaml file
        model_size: 'n', 's', 'm', 'l', 'x'
        epochs: Number of training epochs
        imgsz: Target image size for training
        batch_size: Batch size
    """
    # Initialize from a pre-trained base model
    base_model = f'yolov8{model_size}.pt'
    logger.info(f"Loading pre-trained base model: {base_model}")
    
    model = YOLO(base_model)
    
    logger.info(f"Starting training on dataset {data_yaml} for {epochs} epochs...")
    
    # Train the model
    # Results will be saved to runs/detect/train/ by default
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device='0', # Uses GPU 0 if available, else CPU automatically handled by YOLO but '0' forces CUDA if available. Adjust if CPU only.
        project='custom_training',
        name='retail_finetune'
    )
    
    logger.info("Training complete!")
    logger.info(f"Best model saved at: custom_training/retail_finetune/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on retail dataset")
    parser.add_argument('--data', type=str, default='dataset.yaml', help='Path to dataset.yaml')
    parser.add_argument('--model', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'], help='Model size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    
    args = parser.parse_args()
    
    # Simple check for CPU vs GPU gracefully for ultralytics
    import torch
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Training on CPU will be very slow!")
        
    train_model(args.data, args.model, args.epochs, args.imgsz, args.batch)
