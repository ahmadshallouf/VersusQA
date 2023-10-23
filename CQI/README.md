# Comparative Question Answering Project
### Comparative Question Identification Model

[![UHH](https://www.kus.uni-hamburg.de/5572339/uhh-logo-2010-29667bd15f143feeb1ebd96b06334fddfe378e09.png)](https://www.uni-hamburg.de/) -  <a href="https://www.inf.uni-hamburg.de/en/inst/ab/sems/home.html"><img src="https://www.inf.uni-hamburg.de/5546980/lt-logo-640x361-9345df620ffab7a8ce97149b66c2dfc9d3ff429e.png" width="200" height="100" /></a>

**Due to size problems, the model can't be found in this repository but it can be easily recreated, as explained below**

The Comparative Question Identification Model is a transformer-based classification model created for the identification of comparative questions. 

## Dataset

It uses a mix of different datasets, the full dataset can be found in `final_dataset_english.tsv`. The different train, test and validate sets were created from this file, if they are missing they will be created by the scripts. As an overview, the dataset has 9,876 entries, equally divided between comparative and non-comparative.

### Metrics

Multiple pre-trained models were tested. A full list of the models along with the best metrics obtained during the hyperparameter training can be found in the file `best_models_metrics.json`. The following metrics belong to the model distilbert-base-uncased fine-tuned on [SST](https://towardsdatascience.com/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065) found in [Hugging Face 🤗](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english).

- accuracy: 0.9696
- f1: 0.9708
- loss: 0.1257
- precision: 0.9615
- recall: 0.9803

## Training

Before attempting training make sure you have installed all the requirements in `requirements.txt`. If you don't want to report to [WandB](https://wandb.ai/) please comment this out in the Training arguments. If you have CUDA please indicate this in the code to speed up the training. The model to be trained is set to distilbert-base-uncased fine-tuned on [SST](https://towardsdatascience.com/the-stanford-sentiment-treebank-sst-studying-sentiment-analysis-using-nlp-e1a4cad03065).

After these considerations, the model can be trained simply by running this:

```python
python training.py
```

## Demo and API

The demo is easily run once the model is created. It uses Gradio so it can be operated from your Explorer of preference. **With the model once created** you can start up the demo by executing the following:

```python
python demo.py
```

An API was created to access the model through a request. It is in the main file `main.py`.

```python
python main.py
```

The API is based on FastAPI and once run it requires a GET call to the following endpoint:

```
http://127.0.0.1:8000/is_comparative/Hello_World
```

With input or question at the end. It will return a positive or negative answer as JSON.


## Hyperparameter Training

If you wish to test a new pre-trained model you may do so by modifying the file `hyperparameter_training.py` with your desired options and then you can run with the following:

```python
python hyperparameter_training.py
```

