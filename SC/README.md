# Stance Classification Project


[![UHH](https://www.kus.uni-hamburg.de/5572339/uhh-logo-2010-29667bd15f143feeb1ebd96b06334fddfe378e09.png)](https://www.uni-hamburg.de/) -  <a href="https://www.inf.uni-hamburg.de/en/inst/ab/sems/home.html"><img src="https://www.inf.uni-hamburg.de/5546980/lt-logo-640x361-9345df620ffab7a8ce97149b66c2dfc9d3ff429e.png" width="200" height="100" /></a>


## Dataset

The original dataset from [Panchenko et al., 2019](https://aclanthology.org/W19-4516/) is represented by two files: `comparg_train.tsv` and `comparg_test.tsv`. A split has been created using `data_preparation.py`.

## Training

Install all the needed libraries from `requirements.txt`:

```python
pip install -r requirements.txt 
```

If you don't want to report to [WandB](https://wandb.ai/) please comment the respective lines in `train_main.py`.

The model can be trained by executing `train_main.py`:

```python
python train_main.py
```

## Demo and API

Once the model is created, you can run a demo operated by [Gradio](https://www.gradio.app/):

```python
python demo.py
```

An API was created to access the model through a request. It is in the main file `main.py`.

```python
python main.py
```

## Hyperparameter Training

If you want to optimize the hyperparameters of a new model from HuggingFace, please uncomment the respective lines in the `train_main.py` file and run the code.
