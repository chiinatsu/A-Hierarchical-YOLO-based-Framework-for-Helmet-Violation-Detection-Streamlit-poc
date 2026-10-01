"""Streamlit proof of concept: helmet-violation detection with the thesis's three-stage cascade."""
import io, tempfile
from pathlib import Path
import numpy as np, pandas as pd, cv2, streamlit as st
from pipeline import HelmetPipeline, load_config, draw, SMALL_HEAD_PX

ROOT = Path(__file__).parent
WEIGHTS = {'stage1': ROOT / 'models/stage1_yolo11s-rf8_1280_s0.pt',
           'stage2': ROOT / 'models/stage2_yolov8s_rf3_s0.pt',
           'stage3': ROOT / 'models/stage3_yolo11s_rf6r15_s2.pt'}

st.set_page_config(page_title='Helmet Violation Detection (POC)', layout='wide')
st.title('Helmet Violation Detection: proof of concept')
st.caption('Three-stage YOLO cascade: motorcycle detection (YOLO11s) → rider keypoints and association '
           '(YOLOv8s-pose) → helmet-compliance classification (YOLO11s-cls). Research prototype: '
           'every flag needs human review before any enforcement action.')

missing = [str(p.relative_to(ROOT)) for p in WEIGHTS.values() if not p.exists()]
if missing or not (ROOT / 'config/e2e_final_config_s123.json').exists():
    st.error('Missing files. Put the three weights in models/ and the four JSON files in config/ (see README).\n\n'
             + '\n'.join(f'- {m}' for m in missing)); st.stop()

@st.cache_resource(show_spinner='Loading the three models…')
def get_pipeline():
    cfg = load_config(ROOT / 'config')
    return HelmetPipeline(cfg, {k: str(v) for k, v in WEIGHTS.items()}), cfg

pipe, CFG = get_pipeline()

with st.sidebar:
    st.header('Settings')
    st.markdown(f"**Thesis settings** (final system)  \nStage 1 confidence **{CFG['s1_conf']}** · input {CFG['s1_imgsz']} px  \n"
                f"Stage 2 confidence {CFG['s2_conf']}, NMS IoU {CFG['s2_iou']}  \n"
                f"Anchor **{CFG['anchor']}**, gate T {CFG['T']}  \nStage 3 threshold **{CFG['s3_threshold']}**")
    explore = st.toggle('Change thresholds (exploration only)', value=False)
    s1_conf = st.slider('Stage 1 confidence', 0.05, 0.95, CFG['s1_conf'], 0.01, disabled=not explore)
    s3_thr = st.slider('Stage 3 non-compliant threshold', 0.01, 0.99, CFG['s3_threshold'], 0.01, disabled=not explore)
    if not explore:
        s1_conf, s3_thr = CFG['s1_conf'], CFG['s3_threshold']
    show_riders = st.checkbox('Show rider boxes', value=True)
    st.markdown('---')
    st.markdown('**Legend**  \n🟦 motorcycle · ⬜ rider  \n🟥 **NC** non-compliant · 🟩 **OK** compliant  \n'
                f'number = non-compliant probability  \n(*) = head under {SMALL_HEAD_PX} px (less reliable; '
                'most false accusations in the thesis came from these)')

def heads_table(out):
    rows = [dict(motorcycle=h['moto'] + 1, role=h['role'], decision=h['label'], p_non_compliant=round(h['p_nc'], 3),
                 head_px=int(h['head_px']), small_head=h['small'], association_path=h['path']) for h in out['heads']]
    return pd.DataFrame(rows)

def to_png(img_bgr):
    ok, buf = cv2.imencode('.png', img_bgr); return buf.tobytes()

tab_img, tab_vid, tab_about = st.tabs(['Image', 'Video', 'About'])

with tab_img:
    files = st.file_uploader('Upload one or more CCTV frames', type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
    for f in files or []:
        frame = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.warning(f'{f.name}: could not read the image'); continue
        with st.spinner(f'Running the cascade on {f.name}…'):
            out = pipe.run(frame, s1_conf=s1_conf, s3_threshold=s3_thr)
        ann = draw(frame, out, show_riders)
        nc = sum(h['label'] == 'non-compliant' for h in out['heads'])
        st.subheader(f.name)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Motorcycles', len(out['motos'])); c2.metric('Riders judged', len(out['heads']))
        c3.metric('Flagged non-compliant', nc); c4.metric('Processing time', f"{out['ms']['total']:.0f} ms")
        st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), width='stretch')
        df = heads_table(out)
        if len(df):
            st.dataframe(df, hide_index=True, width='stretch')
        if out['skipped']:
            st.caption(f"{len(out['skipped'])} motorcycle(s) too small to crop (< 24 px) were skipped.")
        st.caption('Time per stage (ms): ' + ', '.join(f'{k} {v:.0f}' for k, v in out['ms'].items() if k != 'total'))
        d1, d2 = st.columns(2)
        d1.download_button('Download annotated image', to_png(ann), f'{Path(f.name).stem}_annotated.png', 'image/png', key=f'i{f.name}')
        d2.download_button('Download results (CSV)', df.to_csv(index=False), f'{Path(f.name).stem}_results.csv', 'text/csv', key=f'c{f.name}')

with tab_vid:
    st.caption('Frames are judged one at a time (no tracking), so the same rider can be flagged in several frames. '
               'On a CPU-only server, expect about 1–3 s per processed frame.')
    vf = st.file_uploader('Upload a short CCTV clip', type=['mp4', 'avi', 'mov', 'mkv'])
    every = st.number_input('Process every Nth frame', 1, 60, 10)
    max_frames = st.number_input('Maximum frames to process', 1, 500, 60)
    if vf and st.button('Run on video'):
        import imageio.v2 as imageio
        tmp_in = tempfile.NamedTemporaryFile(suffix=Path(vf.name).suffix, delete=False); tmp_in.write(vf.read()); tmp_in.close()
        cap = cv2.VideoCapture(tmp_in.name); fps = cap.get(cv2.CAP_PROP_FPS) or 25
        tmp_out = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        writer = imageio.get_writer(tmp_out, fps=max(1, fps / every), codec='libx264', macro_block_size=1)
        rows, idx, done = [], 0, 0
        bar = st.progress(0.0, text='Processing…'); preview = st.empty()
        while done < max_frames:
            ok, frame = cap.read()
            if not ok: break
            if idx % every == 0:
                out = pipe.run(frame, s1_conf=s1_conf, s3_threshold=s3_thr)
                ann = draw(frame, out, show_riders)
                writer.append_data(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB))
                for h in out['heads']:
                    rows.append(dict(frame=idx, time_s=round(idx / fps, 2), motorcycle=h['moto'] + 1, role=h['role'],
                                     decision=h['label'], p_non_compliant=round(h['p_nc'], 3), head_px=int(h['head_px'])))
                done += 1; bar.progress(done / max_frames, text=f'Processed {done} frames')
                if done % 5 == 1: preview.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB), caption=f'frame {idx}', width='stretch')
            idx += 1
        cap.release(); writer.close(); bar.empty()
        df = pd.DataFrame(rows)
        st.success(f'{done} frames processed; {int((df.decision == "non-compliant").sum()) if len(df) else 0} '
                   'non-compliant flags (counted per frame, not per rider).')
        st.video(tmp_out)
        if len(df): st.dataframe(df, hide_index=True, width='stretch')
        st.download_button('Download annotated video', open(tmp_out, 'rb').read(), 'annotated.mp4', 'video/mp4')
        st.download_button('Download per-frame results (CSV)', df.to_csv(index=False), 'video_results.csv', 'text/csv')

with tab_about:
    n = CFG['names']
    st.markdown(f"""
**Models:** Stage 1 `{n['stage1']}` · Stage 2 `{n['stage2']}` · Stage 3 `{n['stage3']}`

**How a frame is processed**
1. **Stage 1** finds every motorcycle (YOLO11s at {CFG['s1_imgsz']} px).
2. Each motorcycle is cropped with a 15% margin and resized to 640 × 640.
3. **Stage 2** finds the riders on it and their five keypoints (neck, shoulders, head, pelvis). Riders whose
   **{CFG['anchor']}** keypoint confidence is below {CFG['T']} are dropped; the rest are ordered driver → passenger.
4. A square head box is built from the {CFG['anchor']} keypoint, shifted by a calibrated offset and sized relative to the rider's width.
5. **Stage 3** classifies the head box; it is flagged **non-compliant** when the probability is at least {CFG['s3_threshold']}.

**Thesis test results (all 234 end-to-end test frames):** 21 of 37 violations detected (recall 0.568), strict F1 0.382,
15 false accusations. Heads under 32 px caused a disproportionate share of false accusations.

**Limitations:** daytime footage from two Metro Manila barangays; no tracking between frames; helmet certification
(PS/ICC) and strap fastening are not checked; religious headwear is not modelled. This is a research prototype, not an
enforcement tool.
""")
