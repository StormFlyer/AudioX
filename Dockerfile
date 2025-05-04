FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy application code 
COPY . .

# Create conda environment
RUN conda create -n AudioX python=3.8.20 -y && \
    conda init bash && \
    echo "conda activate AudioX" >> ~/.bashrc

# Install dependencies with conda activated
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate AudioX && \
    pip install git+https://github.com/ZeyueT/AudioX.git && \
    pip install gradio && \
    conda install -c conda-forge ffmpeg libsndfile -y

# Download model files
RUN mkdir -p model && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json

# Create gradio app file
RUN echo '#!/usr/bin/env python3
import torch
import gradio as gr
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.data.utils import read_video

# Initialize model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, model_config = get_pretrained_model("HKUSTAudio/AudioX")
model = model.to(device)
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]
target_fps = model_config.get("video_fps", 10)

def generate_audio(text_prompt, video_file=None, seconds_total=10):
    conditioning = [{
        "text_prompt": text_prompt,
        "video_prompt": None if video_file is None else [read_video(video_file.name, seek_time=0, duration=seconds_total, target_fps=target_fps).unsqueeze(0)],
        "seconds_start": 0,
        "seconds_total": seconds_total
    }]
    
    audio = generate_diffusion_cond(
        model=model,
        steps=50,
        cfg_scale=7.0,
        conditioning=conditioning,
        sample_size=sample_size,
        device=device
    )
    
    audio = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).cpu().numpy()
    return (sample_rate, audio)

# Create Gradio interface
demo = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Text Prompt", placeholder="Enter a description for the audio to generate..."),
        gr.Video(label="Input Video (Optional)"),
        gr.Slider(minimum=5, maximum=30, value=10, step=5, label="Duration (seconds)")
    ],
    outputs=gr.Audio(label="Generated Audio"),
    title="AudioX: Diffusion Transformer for Anything-to-Audio Generation",
    description="Generate audio or music from text and video inputs using AudioX.",
    examples=[
        ["Peaceful piano music with gentle strings", None, 10],
        ["Birds chirping in a forest", None, 10],
        ["Upbeat electronic dance music with synths", None, 10]
    ]
)

# Launch the interface
demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=True)
' > /app/run_gradio.py && \
   chmod +x /app/run_gradio.py

# Expose the port that Gradio runs on
EXPOSE 7860

# Command to run - directly execute the Python file with the proper environment
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && python /app/run_gradio.py"]
