from flask import Flask, request, jsonify, abort, Response, stream_with_context
from llama_cpp import Llama
import os
import time
from faster_whisper import WhisperModel

app = Flask(__name__)

# ASR model setup
asr_model = WhisperModel("small", device="cpu", compute_type="int8")

# LLM setup
llm_checkpoint = "LLM/tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf"
llm = Llama(model_path=llm_checkpoint, chat_format="llama-2")


@app.route('/answer', methods=['POST'])
def answer():
    data = request.json

    if 'prompt' not in data:
        abort(400, description="Please provide a 'prompt' in the request body")

    prompt = data['prompt']

    try:
        def generate_response():
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an assistant."},
                    {"role": "user", "content": prompt}
                ],
                stop=["</s>"],
                stream=True,
                max_tokens=1024,
                temperature=0.5
            )

            for chunk in output:
                delta = chunk['choices'][0]['delta']
                if 'role' in delta:
                    yield f"{delta['role']}: "
                elif 'content' in delta:
                    yield f"{delta['content']}"

        return Response(stream_with_context(generate_response()), content_type='text/plain')

    except Exception as e:
        abort(500, description=str(e))


@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.json

    if 'file_name' not in data:
        abort(400, description="Please provide a 'file_name' in the request body")

    file_name = data['file_name']

    if not os.path.exists(file_name):
        abort(404, description="File not found")

    try:
        start = time.time()
        segments, info = asr_model.transcribe(file_name, language="en")
        segments = list(segments)
        inference_time = time.time() - start
        transcriptions = " ".join([s.text for s in segments])

        return jsonify({
            "file_name": file_name,
            "transcription": transcriptions,
            "inference_time": inference_time
        })

    except Exception as e:
        abort(500, description=str(e))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)