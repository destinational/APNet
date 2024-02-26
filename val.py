import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('best.pt')
    model.val(data='dataset/VisDrone.yaml',
              split='val',
              imgsz=640,
              batch=4,
              project='',
              name='',
              )
