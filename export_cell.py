# DEPLOY.1 Collect everything the Streamlit app needs into one zip on Drive (no training).
# Run after 0.1–0.3 only (needs ROOT, RES, load_json). Output: MyDrive/helmet_thesis_v2/deploy/helmet_poc_files.zip
import shutil, os, json
S1 = load_json(f'{RES}/stage1_selected_rf8.json'); S2 = load_json(f'{RES}/stage2_selected_rf3.json')
S3 = load_json(f'{RES}/stage3_selected_rf6.json'); E2E = load_json(f'{RES}/e2e_final_config_s123.json')
assert S1 and S2 and S3 and E2E, 'a selection/config file is missing in results/'
OUT = f'{ROOT}/deploy/helmet_poc_files'
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(f'{OUT}/models'); os.makedirs(f'{OUT}/config')
for src, dst in [(S1['weights'], 'stage1_yolo11s-rf8_1280_s0.pt'),
                 (S2['weights'], 'stage2_yolov8s_rf3_s0.pt'),
                 (S3['weights'], 'stage3_yolo11s_rf6r15_s2.pt')]:     # Stage 3 = the epoch-43 checkpoint chosen by F.3
    shutil.copy(src, f'{OUT}/models/{dst}'); print(f'{dst}: {os.path.getsize(src) / 1e6:.1f} MB  <- {src}')
for f in ['stage1_selected_rf8.json', 'stage2_selected_rf3.json', 'stage3_selected_rf6.json', 'e2e_final_config_s123.json']:
    shutil.copy(f'{RES}/{f}', f'{OUT}/config/{f}')
print('Final config:', {k: E2E[k] for k in ['anchor', 'T', 's3_threshold', 'head_off', 'head_scale']})
print('Stage 1 conf/imgsz:', S1['conf'], S1['imgsz'], '| Stage 2 conf/iou:', S2['conf'], S2['iou'])
zp = shutil.make_archive(OUT, 'zip', OUT)
print('Saved', zp, f'({os.path.getsize(zp) / 1e6:.1f} MB). Download it and unzip into the helmet_poc folder.')
