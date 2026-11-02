#!/usr/bin/env python3
# train.py: The stump / log / bag detector (Ultralytics YOLO) on the collected data
import argparse
import os
import shutil

os.environ['YOLO_AUTOINSTALL'] = 'false'
from ultralytics import YOLO

NAMES = ['stump', 'log', 'bag']


def write_yaml(path, train, val):
    with open(path, 'w') as fh:
        fh.write('train:\n' + ''.join(f'  - {t}\n' for t in train))
        fh.write('val:\n' + ''.join(f'  - {v}\n' for v in val))
        fh.write('names:\n' + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))


def main():
    home = os.path.expanduser('~/rs1_18_ws/eval/yolo')
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dirs', nargs='+', default=[f'{home}/ds_showcase', f'{home}/ds_dense'])
    ap.add_argument('--test', default=f'{home}/ds_test_simple')
    ap.add_argument('--model', default='yolo11n.pt', help='pretrained start (COCO)')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--imgsz', type=int, default=736)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--name', default='evidence_v1')
    ap.add_argument('--out', default=f'{home}/runs')
    a = ap.parse_args()
    a.data_dirs = [os.path.abspath(d) for d in a.data_dirs]
    a.test = os.path.abspath(a.test) if a.test else ''

    data = f'{home}/{a.name}_data.yaml'
    write_yaml(data, [f'{d}/images/train' for d in a.data_dirs],
               [f'{d}/images/val' for d in a.data_dirs])
    model = YOLO(f'{home}/weights/{a.model}' if os.path.exists(f'{home}/weights/{a.model}')
                 else a.model)
    model.train(data=data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.out,
                name=a.name, exist_ok=True, flipud=0.0, fliplr=0.5, patience=20,
                workers=8, seed=0, plots=True)
    best = f'{a.out}/{a.name}/weights/best.pt'
    model = YOLO(best)
    model.export(format='onnx', imgsz=a.imgsz, dynamic=False, simplify=True)

    if a.test and os.path.isdir(f'{a.test}/images/val'):
        tdata = f'{home}/{a.name}_test.yaml'
        write_yaml(tdata, [f'{a.test}/images/val'], [f'{a.test}/images/val'])
        m = model.val(data=tdata, imgsz=a.imgsz, split='val', project=a.out,
                      name=f'{a.name}_test', exist_ok=True, plots=True)
        print('TEST (unseen world) mAP50 %.3f mAP50-95 %.3f' % (m.box.map50, m.box.map))
        for i, n in enumerate(NAMES):
            print(f'  {n}: P {m.box.p[i]:.3f} R {m.box.r[i]:.3f} mAP50 {m.box.ap50[i]:.3f}')
    os.makedirs(f'{home}/weights', exist_ok=True)
    shutil.copy(best, f'{home}/weights/{a.name}.pt')
    onnx = best.replace('.pt', '.onnx')
    if os.path.exists(onnx):
        shutil.copy(onnx, f'{home}/weights/{a.name}.onnx')
    print('weights:', f'{home}/weights/{a.name}.pt')


if __name__ == '__main__':
    main()
