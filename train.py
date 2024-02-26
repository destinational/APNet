import warnings
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('ultralytics/cfg/models/APNet.yaml')
    model.train(data='dataset/VisDrone.yaml',
                cache=False,
                imgsz=640,
                epochs=200,
                batch=4,
                workers=2,
                device='',
                # resume='last.pt',
                project='',
                name='',
                )
