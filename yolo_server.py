from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import subprocess
import tempfile
import json
import os
import io
import csv
from pathlib import Path

app = Flask(__name__)
CORS(app)

PYTHON_BIN  = "/opt/anaconda3/envs/yolo/bin/python"
PREDICT_PY  = str(Path(__file__).parent / "predict.py")
MODEL_PATH  = "/Users/jamalmbaye/YOLOX/Yolo_Grain_model/grain_detection_yolom.pt"

print(f"✅ YOLO Server prêt — modèle : {MODEL_PATH}")

# Modèle chargé paresseusement et réutilisé pour le batch (évite de recharger à chaque image)
_MODEL = None
def get_model():
    global _MODEL
    if _MODEL is None:
        from ultralytics import YOLO
        _MODEL = YOLO(MODEL_PATH)
    return _MODEL


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data       = request.json
        image_b64  = data.get('imageBase64')
        conf       = float(data.get('conf', 0.25))
        max_det    = int(data.get('max_det', 10000))
        slope      = float(data.get('slope', 0.026))

        # Sauvegarder l'image dans un dossier temporaire
        out_dir = tempfile.mkdtemp()
        img_path = os.path.join(out_dir, 'input.jpg')
        with open(img_path, 'wb') as f:
            f.write(base64.b64decode(image_b64))

        # Appeler predict.py exactement comme l'app R
        args = [
            PYTHON_BIN, PREDICT_PY,
            '--image',   img_path,
            '--weights', MODEL_PATH,
            '--output',  out_dir,
            '--conf',    str(conf),
            '--max_det', str(max_det)
        ]
        result = subprocess.run(args, capture_output=True, text=True)

        if result.returncode != 0:
            return jsonify({'success': False, 'error': result.stderr}), 500

        # Lire result.json généré par predict.py
        result_json = os.path.join(out_dir, 'result.json')
        with open(result_json) as f:
            res = json.load(f)

        # Lire l'image annotée (boîtes bleues propres)
        annotated_path = os.path.join(out_dir, 'annotated.jpg')
        with open(annotated_path, 'rb') as f:
            annotated_b64 = base64.b64encode(f.read()).decode('utf-8')

        # Calcul masse
        mass_g = round(res['count'] * slope, 2)

        return jsonify({
            'success':         True,
            'annotated_image': annotated_b64,
            'count':           res['count'],
            'mass_g':          mass_g,
            'slope_used':      slope,
            'conf_mean':       res['conf_mean'],
            'conf_min':        res['conf_min'],
            'conf_max':        res['conf_max'],
            'class_counts':    res['class_counts']
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/detect-batch', methods=['POST'])
def detect_batch():
    """
    Traitement par lot.
    Entrée JSON : { "images": [{"name": "...", "data": "<base64>"}, ...],
                    "conf": 0.25, "max_det": 10000, "slope": 0.026, "face_factor": 2.0 }
    Sortie JSON : { "success": true, "rows": [...], "csv": "<texte csv>" }
    Le modèle est chargé une seule fois en mémoire pour tout le lot.
    """
    try:
        data        = request.json
        images      = data.get('images', [])
        conf        = float(data.get('conf', 0.25))
        max_det     = int(data.get('max_det', 10000))
        slope       = float(data.get('slope', 0.026))
        face_factor = float(data.get('face_factor', 2.0))

        if not images:
            return jsonify({'success': False, 'error': 'Aucune image reçue'}), 400

        model = get_model()
        names = model.names
        rows  = []

        for item in images:
            img_name = item.get('name', 'image')
            img_b64  = item.get('data')
            row = {
                'image_id':     img_name,
                'count':        0,
                'grains_total': 0,
                'mass_g':       0.0,
                'conf_mean':    0.0,
                'conf_min':     0.0,
                'conf_max':     0.0,
                'class_counts': '',
                'status':       'ok',
            }

            if not img_b64:
                row['status'] = 'error: image vide'
                rows.append(row)
                continue

            try:
                # Écriture temporaire de l'image
                tmp_dir = tempfile.mkdtemp()
                img_path = os.path.join(tmp_dir, img_name)
                with open(img_path, 'wb') as f:
                    f.write(base64.b64decode(img_b64))

                # Même logique de comptage que predict.py
                results = model(img_path, conf=conf, max_det=max_det, iou=0.5, verbose=False)
                result  = results[0]
                count   = len(result.boxes)

                # Comptage par classe
                cc = {}
                confs = []
                if result.boxes is not None and len(result.boxes) > 0:
                    for cls_id in result.boxes.cls.tolist():
                        label = names[int(cls_id)]
                        cc[label] = cc.get(label, 0) + 1
                    confs = [float(c) for c in result.boxes.conf.tolist()]

                grains_total = round(count * face_factor)
                mass_g       = round(grains_total * slope, 2)

                row.update({
                    'count':        count,
                    'grains_total': grains_total,
                    'mass_g':       mass_g,
                    'conf_mean':    round(sum(confs) / len(confs), 4) if confs else 0.0,
                    'conf_min':     round(min(confs), 4) if confs else 0.0,
                    'conf_max':     round(max(confs), 4) if confs else 0.0,
                    'class_counts': ';'.join(f'{k}:{v}' for k, v in cc.items()),
                })
            except Exception as e:
                row['status'] = f'error: {e}'

            rows.append(row)

        # Construction du CSV en mémoire
        fieldnames = ['image_id', 'count', 'grains_total', 'mass_g',
                      'conf_mean', 'conf_min', 'conf_max', 'class_counts', 'status']
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        csv_text = buf.getvalue()

        return jsonify({
            'success': True,
            'rows':    rows,
            'csv':     csv_text,
            'params':  {'conf': conf, 'max_det': max_det, 'slope': slope, 'face_factor': face_factor},
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    print("🚀 YOLO Server sur http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=False)
