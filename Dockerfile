FROM continuumio/miniconda3

WORKDIR /app

COPY . /app

RUN conda install -c conda-forge rdkit -y

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]