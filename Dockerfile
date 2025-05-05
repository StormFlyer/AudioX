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

# Make sure run_gradio.py is executable
RUN chmod +x /app/run_gradio.py

# Expose the port that Gradio runs on
EXPOSE 7860

# Run the application directly
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && export GRADIO_SERVER_NAME=0.0.0.0 && export GRADIO_SERVER_PORT=7860 && python /app/run_gradio.py --model-config /app/model/config.json --ckpt-path /app/model/model.ckpt"]
