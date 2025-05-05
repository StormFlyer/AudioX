FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy application code 
COPY . .

# Create conda environment and initialize
RUN conda create -n AudioX python=3.8.20 -y && \
    conda init bash && \
    echo "conda activate AudioX" >> ~/.bashrc

# Install dependencies with properly activated conda environment
SHELL ["/bin/bash", "--login", "-c"]
RUN conda activate AudioX && \
    pip install git+https://github.com/ZeyueT/AudioX.git && \
    pip install gradio && \
    conda install -c conda-forge ffmpeg libsndfile -y

# Download model files
RUN mkdir -p model && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json

# Expose the port that Gradio runs on
EXPOSE 7860

# Keep container running even if application fails
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && (python /app/run_gradio.py || (echo 'Application failed to start, keeping container alive for debugging' && tail -f /dev/null))"]
