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

# Create simple wrapper script
RUN echo '#!/bin/bash\n\
export GRADIO_SERVER_NAME="0.0.0.0"\n\
export GRADIO_SERVER_PORT=7860\n\
python /app/run_gradio.py --model-config /app/model/config.json\n\
if [ $? -ne 0 ]; then\n\
  echo "Application failed to start, keeping container alive for debugging"\n\
  tail -f /dev/null\n\
fi\n' > /app/start.sh && \
    chmod +x /app/start.sh

# Expose the port that Gradio runs on
EXPOSE 7860

# Run the application
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && /app/start.sh"]
