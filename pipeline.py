"""Inference-only copy of the thesis pipeline (Thesis_Pipeline_v2_Final_v1_9.ipynb, cells 3.1, 3.4a, 5.1, RQ2 B).
Stage 1 YOLO11s motorcycle detector -> Stage 2 YOLOv8s-pose riders + keypoints -> head box from the anchor
keypoint (calibrated offset and scale) -> Stage 3 YOLO11s-cls compliant / non-compliant.
Every constant below must match the notebook; do not change them unless the models are retrained."""
import json, time
from pathlib import Path
import numpy as np
import cv2
from ultralytics import YOLO

# ---- constants from the notebook (cells 3.1 and 5.1) ----
KEYPOINT_NAMES = ['neck', 'shoulder_l', 'head', 'shoulder_r', 'pelvis']   # order in the Roboflow export
TORSO_KPT_NAMES = ['neck', 'shoulder_l', 'shoulder_r', 'pelvis']
INFERIOR_KPT_NAMES = ['pelvis']
PAD, S2_SIZE, MIN_CROP_PX = 0.15, 640, 24     # Stage 2 crop: 15% margin, square 640 stretch, skip < 24 px
S3_PAD = 0.10                                 # margin added around the reconstructed head box
SMALL_HEAD_PX = 32                            # COCO small-object boundary (thesis Section 5.8)


def load_config(config_dir):
    """Reads the four result files saved on Drive by the final notebook."""
    d = Path(config_dir)
    rd = lambda n: json.loads((d / n).read_text())
    e2e, s1, s2, s3 = (rd('e2e_final_config_s123.json'), rd('stage1_selected_rf8.json'),
                       rd('stage2_selected_rf3.json'), rd('stage3_selected_rf6.json'))
    return dict(anchor=e2e['anchor'], T=float(e2e['T']), s3_threshold=float(e2e['s3_threshold']),
                head_off=tuple(e2e['head_off']), head_scale=float(e2e['head_scale']),
                s1_imgsz=int(s1['imgsz']), s1_conf=float(s1['conf']),
                s2_conf=float(s2['conf']), s2_iou=float(s2['iou']),
                names=dict(stage1=s1.get('run'), stage2=s2.get('run'), stage3=s3.get('name')))


def _gated_mean(xy, c, idx, T):
    keep = [i for i in idx if c[i] >= T]
    return xy[keep].mean(0) if keep else None


def _roles_by_y(ys):
    """Highest on screen = Passenger (rear), lowest = Driver, others Middle (thesis Section 3.5.2)."""
    order, n, roles = np.argsort(ys), len(ys), [None] * len(ys)
    for rank, i in enumerate(order):
        roles[i] = 'Driver' if rank == n - 1 else ('Passenger' if rank == 0 else f'Middle{rank}')
    return roles


class HelmetPipeline:
    def __init__(self, cfg, weights, device=None):
        self.cfg = cfg
        self.det = YOLO(weights['stage1'], task='detect')
        self.pose = YOLO(weights['stage2'], task='pose')
        self.cls = YOLO(weights['stage3'], task='classify')
        nc = [k for k, v in self.cls.names.items() if v == 'non-compliant']
        assert nc, f'Stage 3 model has no "non-compliant" class: {self.cls.names}'
        self.nc_idx = nc[0]
        self.device = device
        k = lambda names: [KEYPOINT_NAMES.index(n) for n in names]
        self.head_kpts = k([cfg['anchor']])
        self.torso_kpts, self.inf_kpts = k(TORSO_KPT_NAMES), k(INFERIOR_KPT_NAMES)

    # ---- Stage 2 helpers (cell 3.4a) ----
    def _associate(self, boxes, kxy, kc, T):
        riders = []
        for i in range(len(boxes)):
            head = _gated_mean(kxy[i], kc[i], self.head_kpts, T)
            if head is None:
                continue                                           # gate: anchor keypoint below T -> rider dropped
            torso = _gated_mean(kxy[i], kc[i], self.torso_kpts, T)
            inf_ok = any(kc[i][j] >= T for j in self.inf_kpts)
            riders.append(dict(i=i, head=head, torso=torso, inf_ok=inf_ok))
        if not riders:
            return 'none', [], [], np.zeros((0, 2))
        path = 'A' if all(r['inf_ok'] and r['torso'] is not None for r in riders) else 'B'
        ys = [r['torso'][1] if path == 'A' else r['head'][1] for r in riders]
        return path, [r['i'] for r in riders], _roles_by_y(np.array(ys)), np.array([r['head'] for r in riders])

    @staticmethod
    def _make_crop(frame, box):
        H, W = frame.shape[:2]; x1, y1, x2, y2 = [int(v) for v in box]
        px, py = int((x2 - x1) * PAD), int((y2 - y1) * PAD)
        cx1, cy1, cx2, cy2 = max(0, x1 - px), max(0, y1 - py), min(W, x2 + px), min(H, y2 + py)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or min(crop.shape[:2]) < MIN_CROP_PX:
            return None, (cx1, cy1, cx2, cy2)
        interp = cv2.INTER_LINEAR if S2_SIZE / max(cx2 - cx1, cy2 - cy1) >= 1 else cv2.INTER_AREA
        return cv2.resize(crop, (S2_SIZE, S2_SIZE), interpolation=interp), (cx1, cy1, cx2, cy2)

    def _predict(self, model, imgs, batch, **kw):
        out = []
        for i in range(0, len(imgs), batch):
            out += list(model.predict(imgs[i:i + batch], verbose=False, device=self.device, **kw))
        return out

    # ---- full pipeline (cell 5.1 run_pipeline) ----
    def run(self, frame, s1_conf=None, s3_threshold=None):
        c = self.cfg
        s1_conf = c['s1_conf'] if s1_conf is None else s1_conf
        s3_thr = c['s3_threshold'] if s3_threshold is None else s3_threshold
        H, W = frame.shape[:2]; t = {}
        t0 = time.perf_counter()
        r1 = self.det.predict(frame, imgsz=c['s1_imgsz'], conf=s1_conf, verbose=False, device=self.device)[0]
        motos = r1.boxes.xyxy.cpu().numpy(); moto_conf = r1.boxes.conf.cpu().numpy()
        t['stage1'] = time.perf_counter() - t0; t0 = time.perf_counter()
        crops, regions, skipped = [], [], []
        for b in motos:
            cr, reg = self._make_crop(frame, b)
            (skipped if cr is None else regions).append(reg)
            if cr is not None: crops.append(cr)
        r2 = self._predict(self.pose, crops, 64, imgsz=S2_SIZE, conf=c['s2_conf'], iou=c['s2_iou']) if crops else []
        t['stage2'] = time.perf_counter() - t0; t0 = time.perf_counter()
        heads = []
        for mi, (res, (cx1, cy1, cx2, cy2)) in enumerate(zip(r2, regions)):
            sx, sy = (cx2 - cx1) / S2_SIZE, (cy2 - cy1) / S2_SIZE
            if res.boxes is None or len(res.boxes) == 0 or res.keypoints is None:
                continue
            b = res.boxes.xyxy.cpu().numpy(); kxy = res.keypoints.xy.cpu().numpy()
            kc = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else np.ones(kxy.shape[:2])
            path, keep, roles, hpts = self._associate(b, kxy, kc, c['T'])
            for k, i in enumerate(keep):
                bx = b[i]; rw = (bx[2] - bx[0]) * sx
                kx, ky = cx1 + hpts[k][0] * sx, cy1 + hpts[k][1] * sy                     # anchor keypoint
                hx, hy = kx + c['head_off'][0] * rw, ky + c['head_off'][1] * rw          # calibrated head centre
                side = c['head_scale'] * rw
                heads.append(dict(moto=mi, role=roles[k], path=path, kpt=(kx, ky), center=(hx, hy),
                                  rider_box=[cx1 + bx[0] * sx, cy1 + bx[1] * sy, cx1 + bx[2] * sx, cy1 + bx[3] * sy],
                                  box=[max(0, hx - side / 2), max(0, hy - side / 2),
                                       min(W, hx + side / 2), min(H, hy + side / 2)], head_px=side))
        t['assoc'] = time.perf_counter() - t0; t0 = time.perf_counter()
        hcrops = []
        for h in heads:
            x1, y1, x2, y2 = h['box']; p = S3_PAD * (x2 - x1)
            hc = frame[int(max(0, y1 - p)):int(min(H, y2 + p)), int(max(0, x1 - p)):int(min(W, x2 + p))]
            hcrops.append(hc if hc.size else np.zeros((8, 8, 3), np.uint8))
        r3 = self._predict(self.cls, hcrops, 128, imgsz=224) if hcrops else []
        for h, r in zip(heads, r3):
            h['p_nc'] = float(r.probs.data[self.nc_idx])
            h['label'] = 'non-compliant' if h['p_nc'] >= s3_thr else 'compliant'
            h['small'] = h['head_px'] < SMALL_HEAD_PX
        t['stage3'] = time.perf_counter() - t0
        t['total'] = sum(t.values())
        return dict(motos=motos, moto_conf=moto_conf, regions=regions, skipped=skipped, heads=heads,
                    ms={k: v * 1000 for k, v in t.items()})


# ---- drawing ----
GREEN, RED, BLUE, GREY = (40, 170, 60), (40, 40, 220), (220, 140, 30), (160, 160, 160)

def draw(frame, out, show_riders=True):
    img = frame.copy()
    lw = max(2, int(round(img.shape[1] / 700)))
    fs = max(0.45, img.shape[1] / 2600)
    for (x1, y1, x2, y2), cf in zip(out['motos'], out['moto_conf']):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BLUE, lw)
    for h in out['heads']:
        if show_riders:
            x1, y1, x2, y2 = map(int, h['rider_box']); cv2.rectangle(img, (x1, y1), (x2, y2), GREY, 1)
        col = RED if h['label'] == 'non-compliant' else GREEN
        x1, y1, x2, y2 = map(int, h['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, lw)
        tag = f"{'NC' if h['label'] == 'non-compliant' else 'OK'} {h['p_nc']:.2f}" + (' (*)' if h['small'] else '')
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return img
