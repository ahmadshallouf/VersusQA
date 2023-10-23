# Comparative-Question-Answering-Summarization

[![UHH](https://www.kus.uni-hamburg.de/5572339/uhh-logo-2010-29667bd15f143feeb1ebd96b06334fddfe378e09.png)](https://www.uni-hamburg.de/) -  <a href="https://www.inf.uni-hamburg.de/en/inst/ab/sems/home.html"><img src="https://www.inf.uni-hamburg.de/5546980/lt-logo-640x361-9345df620ffab7a8ce97149b66c2dfc9d3ff429e.png" width="200" height="100" /></a>

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

