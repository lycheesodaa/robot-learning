"""Launch: streamlit run scripts/concept_ui.py --server.address 127.0.0.1"""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import streamlit as st
import streamlit.components.v1 as components
from vla_safety.concepts.archive import RunArchive, read_events, verify_archive
from vla_safety.concepts.runner import run_experiment, validate_config

st.set_page_config(page_title='Concept experiments', layout='wide')
st.title('Concept experiments')
st.caption('Recorded-input studies of fragility and other concepts. Saved replays work without a model.')


class Worker:
    def __init__(self):
        self.pool = ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn'))
        self.future = None
        self.folder = None
        self.lock = threading.Lock()

    def submit(self, config):
        with self.lock:
            if self.future is not None and not self.future.done():
                raise ValueError('An experiment is already running')
            self.folder = RunArchive.new_path(ROOT / 'artifacts/concepts', config['concept']['id'])
            self.future = self.pool.submit(run_experiment, config, ROOT, output=self.folder)


@st.cache_resource
def worker():
    return Worker()


saved, launch = st.tabs(['Saved runs / offline replay', 'Run experiment'])
with saved:
    paths = sorted((ROOT / 'artifacts/concepts').glob('*/*/status.json'), reverse=True)
    choice = st.selectbox('Saved run', [str(p.parent.relative_to(ROOT)) for p in paths], index=0 if paths else None)
    if choice:
        folder = ROOT / choice
        status = json.loads((folder / 'status.json').read_text())
        st.write(status)
        if st.button('Verify archive'):
            st.json(verify_archive(folder))
        report = folder / 'replay.html'
        if report.exists():
            st.download_button('Download standalone replay', report.read_bytes(), 'replay.html', 'text/html')
            components.html(report.read_text(), height=1100, scrolling=True)
        else:
            st.info('No final report yet. Use the replay CLI to export the committed events of an interrupted run.')
with launch:
    study = st.selectbox('Study', ['Fragile-object release', 'General concept contrasts'])
    default_path = ('artifacts/concept_inputs/fragile_release/config.json' if study == 'Fragile-object release'
                    else 'artifacts/concept_inputs/sample/config.json')
    source = st.text_input('Prepared configuration path', default_path, key=f'path:{study}')
    if study == 'Fragile-object release':
        st.caption('Compare fragile/durable objects, high/low release positions, and explicit/implicit release rules. '
                   'The 5 cm default is illustrative. Outcomes need an aligned release trajectory.')
    path = ROOT / source
    initial = path.read_text() if path.is_file() else '{}'
    text = st.text_area('Experiment configuration', initial, height=350, key=f'config:{source}')
    st.caption('The worker keeps one checkpoint loaded between jobs. New labels need a corresponding dataset.')
    if st.button('Run experiment'):
        try:
            config = json.loads(text)
            validate_config(config)
            worker().submit(config)
        except (ValueError, OSError) as exc:
            st.error(str(exc))

    @st.fragment(run_every='1s')
    def progress():
        current = worker()
        if current.folder is None:
            return
        st.write(str(current.folder))
        if current.future.done():
            try:
                st.success(f'Saved: {current.future.result()}')
            except Exception as exc:
                st.error(str(exc))
        else:
            events = read_events(current.folder)
            st.write(events[-1] if events else 'Starting worker…')
            if current.folder.exists() and st.button('Cancel after current prediction'):
                (current.folder / 'CANCEL').touch()
    progress()
