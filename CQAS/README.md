# Comparative-Question-Answering-Summarization

The Comparative Question Answering Summarization is an abstractive text summarization model used for summarizing CQAM (Comparative Question Answering Machine) outputs. 

### Dataset

Models are trained on a small dataset obtained by feeding CQAM outputs to ChatGPT to obtain ground truth summaries. The dataset is then pre-processed by modifying the model input to the form appropriate to HuggingFace models. The dataset consists of 1602 inputs, with 80% dedicated to training, 10% to validation, and 10% to testing.

### Evaluation Metrics

For evaluation, following metrics were used:

- Rouge1
- Rouge2
- RougeL
- RougeLSum

### Training

After installing requirements in `requirements.txt`, you can obtain the best-performing model by running following command:

```python
python train.py
```

Hyperparameter tuning can be done via:

```python
python raytune.py
```

### API

The service can be started with:

```python
uvicorn service:app --reload
```

Summarization is available via endpoint /summary

### Demo

You can try generating summaries via demo:

```python
python demo.py
```

