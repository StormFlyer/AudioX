FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy your application code
COPY . .

# Create conda environment
RUN conda create -n AudioX python=3.8.20 -y && \
    echo "source activate AudioX" > ~/.bashrc

SHELL ["/bin/bash", "--login", "-c"]

# Install dependencies
RUN conda activate AudioX && \
    pip install git+https://github.com/ZeyueT/AudioX.git && \
    conda install -c conda-forge ffmpeg libsndfile -y

# Download model files
RUN mkdir -p model && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/model.ckpt -O model/model.ckpt && \
    wget https://huggingface.co/HKUSTAudio/AudioX/resolve/main/config.json -O model/config.json

# Create startup script
RUN echo '#!/bin/bash\nsource activate AudioX\npython3 run_gradio.py --model-config model/config.json --share' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port that Gradio runs on
EXPOSE 7860

# Run the application
CMD ["/app/start.sh"]
