# Voice Assistant

## Installation
1) Create a virtual environment with python 3.10.
2) Install the dependencies:
```bash
pip install -r requirements.txt
```
3) Download LLM model:
```bash
bash download_llm.sh
```

## Usage
Run API to accept requests for transcription and question answering:
```bash
python3 asr_llm_api.py
```
Run app to record audio and send it to the API:
```bash
python3 vad_pipeline.py
```