# NYC Taxifare Predict designed by LeWagon

# ⛰ "Train At Scale" Unit 🗻

In this unit, you will learn how to package the notebook provided by the Data Science team at WagonCab, and how to scale it so that it can be trained locally on the full dataset.

This unit consists of the 5 challenges below, they are all grouped up in this single `README` file.

Simply follow the guide and `git push` after each main section so we can track your progress!

# 1️⃣ Local Setup

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

As lead ML Engineer for the project, your first role is to set up a local working environment (with `pyenv`) and a python package that only contains the skeleton of your code base.

💡 Packaging notebooks is a key ML Engineer skill. It allows
- other users to collaborate on the code
- you to clone the code locally or on a remote machine to, for example, train the `taxifare` model on a more powerful machine
- you to put the code in production (on a server that never stops running) to expose it as an **API** or through a **website**
- you to render the code operable so that it can be run manually or plugged into an automation workflow

### 1.1) Create a new pyenv called [🐍 taxifare-env]

🐍 Create the virtual env

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
python --version # First, check your Python version for <YOUR_PYTHON_VERSION> below (e.g. 3.10.6)
```

```bash
pyenv virtualenv <YOUR_PYTHON_VERSION> taxifare-env
pyenv local taxifare-env
pip install --upgrade pip
code .
```

Then, make sure both your OS' Terminal and your VS Code's integrated Terminal display `[🐍 taxifare-env]`.
In VS code, open any `.py` file and check that `taxifare-env` is activated by clicking on the pyenv section in the bottom right, as seen below:

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-setup.png" target="_blank">
    <img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-setup.png' width=400>
</a>

### 1.2) Get familiar with the taxifare package structure

❗️Take 10 minutes to understand the structure of the boilerplate we've prepared for you (don't go into detail); its entry point is `taxifare.interface.main_local`: follow it quickly.

```bash
. # Challenge folder root
├── Makefile          # 🚪 Your command "launcher". Use it extensively (launch training, tests, etc...)
├── README.md         # The file you are reading right now!
├── notebooks
│   └── datascientist_deliverable.ipynb   # The deliverable from the DS team!
├── requirements.txt   # List all third-party packages to add to your local environment
├── setup.py           # Enable `pip install` for your package
├── taxifare           # The code logic for this package
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py  # 🚪 Your main Python entry point containing all "routes"
│   └── ml_logic
│   |    ├── __init__.py
│   |    ├── data.py           # Save, load and clean data
│   |    ├── encoders.py       # Custom encoder utilities
│   |    ├── model.py          # TensorFlow model
│   |    ├── preprocessor.py   # Sklearn preprocessing pipelines
│   |    ├── registry.py       # Save and load models
|   ├── utils.py    # # Useful python functions with no dependencies on taxifare logic
|   ├── params.py   # Global project params
|
├── tests  # Tests to run using `make test_...`
│   ├── ...
│   └── ...
├── .gitignore
```

🐍 Install your package on this new virtual env

```bash
cd ~/code/<user.github_nickname>/{{local_path_to("07-ML-Ops/01-Train-at-scale/01-Train-at-scale")}}
pip install -e .
```

Make sure the package is installed by running `pip list | grep taxifare`; it should print the absolute path to the package.


### 1.3) Where is the data?

**Raw data is in Google Big Query**

WagonCab's engineering team stores all it's cab course history since 2009 in a massive Big Query table `wagon-public-datasets.taxifare.raw_all`.
- This table contains `1.1 Million` for this challenge exactly, from **2009 to jun 2015**.
- *(Note from Le Wagon: In reality, there is 55M rows but we limited that for cost-control in the whole module)*

**Check access to Google Cloud Platform**
Your computer should already be configured to have access to Google Cloud Platform since [setup-day](https://github.com/lewagon/data-setup/blob/master/macOS.md#google-cloud-platform-setup)

🧪 Check that everything is fine
```bash
make test_gcp_setup
```

**We'll always cache all intermediate data locally in `~/.lewagon/mlops/` to avoid querying BQ twice**

💾 Let's store our `data` folder *outside* of this challenge folder so that it can be accessed by all other challenges throughout the whole ML Ops module. We don't want it to be tracked by `git` anyway!

``` bash
# Create the data folder
mkdir -p ~/.lewagon/mlops/data/

# Create relevant subfolders
mkdir ~/.lewagon/mlops/data/raw
mkdir ~/.lewagon/mlops/data/processed
```

💡While we are here, let's also create a storage folder for our `training_outputs` that will also be shared by all challenges

```bash
# Create the training_outputs folder
mkdir ~/.lewagon/mlops/training_outputs

# Create relevant subfolders
mkdir ~/.lewagon/mlops/training_outputs/metrics
mkdir ~/.lewagon/mlops/training_outputs/models
mkdir ~/.lewagon/mlops/training_outputs/params
```

You can now see that the data for the challenges to come is stored in `~/.lewagon/mlops/`, along with the notebooks of the Data Science team and the model outputs:

``` bash
tree -a ~/.lewagon/mlops/

# YOU SHOULD SEE THIS
├── data          # This is where you will:
│   ├── processed # Store intermediate, processed data
│   └── raw       # Download samples of the raw data
└── training_outputs
    ├── metrics # Store trained model metrics
    ├── models  # Store trained model weights (can be large!)
    └── params  # Store trained model hyperparameters
```

☝️ Feel free to remove all files but keep this empty folder structure at any time using

```bash
make reset_local_files
```

</details>

# 2️⃣ Understand the Work of a Data Scientist

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Duration:  spend 1 hour on this*

🖥️ Open `datascientist_deliverable.ipynb` with VS Code (forget about Jupyter for this module), and run all cells carefully, while understanding them. This handover between you and the DS team is the perfect time to interact with them (i.e. your buddy or a TA).

❗️Make sure to use `taxifare-env` as an `ipykernel` venv

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-notebook.png" target="_blank">
    <img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/pyenv-notebook.png' width=400>
</a>

</details>


# 3️⃣ Package Your Code

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Your goal is to be able to run the `taxifare.interface.main_local` module as seen below

```bash
# -> model
python -m taxifare.interface.main_local
```

🖥️ To do so, please code the missing parts marked with `# YOUR CODE HERE` in the following files; it should follow the Notebook pretty closely!

```bash
├── taxifare
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py   # 🔵 🚪 Entry point: code both `preprocess_and_train()` and `pred()`
│   └── ml_logic
│       ├── __init__.py
│       ├── data.py          # 🔵 your code here
│       ├── encoders.py      # 🔵 your code here
│       ├── model.py         # 🔵 your code here
│       ├── preprocessor.py  # 🔵 your code here
│       ├── registry.py  # ✅ `save_model` and `load_model` are already coded for you
|   ├── params.py # 🔵 You need to fill your GCP_PROJECT
│   ├── utils.py
```

**🧪 Test your code**

Make sure you have the package installed correctly in your current taxifare-env, if not

```bash
pip list | grep taxifare
```

Then, make sure your package runs properly with `python -m taxifare.interface.main_local`.
- Debug it until it runs!
- Use the following dataset sizes

```python
# taxifare/ml_logic/params.py
DATA_SIZE = '1k'   # To iterate faster in debug mode 🐞
DATA_SIZE = '200k' # Should work at least once
# DATA_SIZE = 'all' 🚨 DON'T TRY YET, it's too big and will cost money!
```

Then, only try to pass tests with `make test_preprocess_and_train`!

✅ When you are all green, track your results on kitt with `make test_kitt`

</details>

# 4️⃣ Investigate Scalability

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

*⏱ Duration:  spend 20 minutes at most on this*

Now that you've managed to make the package work for a small dataset, time to see how it will handle the real dataset!

👉 Change `ml_logic.params.DATA_SIZE` to `all` to start getting serious!

🕵️ Investigate which part of your code takes **the most time** and uses **the most memory**  using `taxifare.utils.simple_time_and_memory_tracker` to decorate the methods of your choice.

```python
# taxifare.ml_logic.data.py
from taxifare.utils import simple_time_and_memory_tracker

@simple_time_and_memory_tracker
def clean_data() -> pd.DataFrame:
    ...
```

💡 If you don't remember exactly how decorators work, refer to our [04/05-Communicate](https://kitt.lewagon.com/camps/<user.batch_slug>/lectures/content/04-Decision-Science_05-Communicate.slides.html?title=Communicate#/6/3) lecture!

🕵️ Try to answer the following questions with your buddy:
- What part of your code holds the key bottlenecks ?
- What kinds of bottlenecks are the most worrying? (time? memory?)
- Do you think it will scale if we had given you the 50M rows ? 500M ? By the way, the [real NYC dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page) is even bigger and weights in at about 156GB!
- Can you think about potential solutions? Write down your ideas, but do not implement them yet!
</details>


# 5️⃣ Incremental Processing

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Your goal is to improve your codebase to be able to train the model on unlimited amount of rows, **without reaching RAM limits**, on a single computer.

## 5.1) Discussion

**What did we learn?**

We have memory and time constraints:
- A `(55M, 8)`-shaped raw data gets loaded into memory as a DataFrame and takes up about 10GB of RAM, which is too much for most computers.
- A `(55M, 65)`-shaped preprocessed DataFrame is even bigger.
- The `ml_logic.encoders.compute_geohash` method takes a very long time to process 🤯

One solution is to pay for a *cloud Virtual Machine (VM)* with enough RAM and process it there (this is often the simplest way to deal with such a problem).

**Proposed solution: incremental preprocessing 🔪 chunk by chunk 🔪**

<img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/process_by_chunk.png" width=500>

💡 As our preprocessor is *stateless*, we can easily:
- Avoid computing any _column-wise statistics_ but only perform _row-by-row preprocessing_
- Decouple the _preprocessing_ from the _training_ and store any intermediate results on disk!

🙏 Therefore, let's do the preprocessing *chunk by chunk*, with chunks of limited size (e.g. 100.000 rows), each chunk fitting nicely in memory:

1. We'll store `data_processed_chunk_01` on a hard-drive.
2. Then append `data_processed_chunk_02` to the first.
3. etc...
4. Until a massive CSV is stored at `~/.lewagon/mlops/data/processed/processed_all.csv`

5. In section 6️⃣, we'll `train()` our model chunk-by-chunk too by loading & training iteratively on each chunk (more on that next section)

## 5.2) Your turn: code `def preprocess()`

👶 **First, let's bring back smaller dataset sizes for debugging purposes**

```python
# params.py
DATA_SIZE = '1k'
CHUNK_SIZE = 200
```

**Then, code the new route given below by `def preprocess()` in your `ml_logic.interface.main_local` module; copy and paste the code below to get started**

[//]: # (  🚨 Code below is NOT the single source of truth. Original is in data-solutions repo 🚨 )

<br>

<details>
  <summary markdown='span'>👇 Code to copy 👇</summary>

```python
def preprocess(min_date: str = '2009-01-01', max_date: str = '2015-01-01') -> None:
    """
    Query and preprocess the raw dataset iteratively (by chunks).
    Then store the newly processed (and raw) data on local hard-drive for later re-use.

    - If raw data already exists on local disk:
        - use `pd.read_csv(..., chunksize=CHUNK_SIZE)`

    - If raw data does not yet exists:
        - use `bigquery.Client().query().result().to_dataframe_iterable()`

    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess by batch" + Style.RESET_ALL)

    from taxifare.ml_logic.data import clean_data
    from taxifare.ml_logic.preprocessor import preprocess_features

    min_date = parse(min_date).strftime('%Y-%m-%d') # e.g '2009-01-01'
    max_date = parse(max_date).strftime('%Y-%m-%d') # e.g '2009-01-01'

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM {GCP_PROJECT_WAGON}.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
        """
    # Retrieve `query` data as dataframe iterable
    data_query_cache_path = Path(LOCAL_DATA_PATH).joinpath("raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv")
    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")

    data_query_cache_exists = data_query_cache_path.is_file()
    if data_query_cache_exists:
        print("Get a dataframe iterable from local CSV...")
        chunks = None
        # YOUR CODE HERE

    else:
        print("Get a dataframe iterable from Querying Big Query server...")
        chunks = None
        # 🎯 Hints: `bigquery.Client(...).query(...).result(page_size=...).to_dataframe_iterable()`
        # YOUR CODE HERE

    for chunk_id, chunk in enumerate(chunks):
        print(f"processing chunk {chunk_id}...")

        # Clean chunk
        # YOUR CODE HERE

        # Create chunk_processed
        # 🎯 Hints: Create (`X_chunk`, `y_chunk`), process only `X_processed_chunk`, then concatenate (X_processed_chunk, y_chunk)
        # YOUR CODE HERE

        # Save and append the processed chunk to a local CSV at "data_processed_path"
        # 🎯 Hints: df.to_csv(mode=...)
        # 🎯 Hints: We want a CSV without index nor headers (they'd be meaningless)
        # YOUR CODE HERE

        # Save and append the raw chunk if not `data_query_cache_exists`
        # YOUR CODE HERE
    print(f"✅ data query saved as {data_query_cache_path}")
    print("✅ preprocess() done")


```

</details>

<br>

**❓Try to create and store the following preprocessed datasets**

- `data/processed/train_processed_1k.csv` by running `preprocess()`

<br>

**🧪 Test your code**

Test your code with `make test_preprocess_by_chunk`.

✅ When you are all green, track your results on kitt with `make test_kitt`

<br>

**❓Finally, create and store the real preprocessed datasets**

Using:
```python
# params.py
DATA_SIZE = 'all'
CHUNK_SIZE = 100000
```

🎉 Given a few hours of computation, we could easily process the 55 Million rows too, but let's not do it today 😅

</details>

# 6️⃣ Incremental Learning

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

<br>

🎯 Goal: train our model on the full `.../processed/processed_all.csv`

## 6.1) Discussion

In theory, we cannot load such a big dataset of shape `(xxMillions, 65)` into RAM all at once, but we can load it in chunks.

**How do we train a model in chunks?**

This is called **incremental learning**, or **partial_fit**
- We initialize a model with random weights ${\theta_0}$
- We load the first `data_processed_chunk` into memory (say, 100_000 rows)
- We train our model on the first chunk and update its weights accordingly ${\theta_0} \rightarrow {\theta_1}$
- We load the second `data_processed_chunk` into memory
- We *retrain* our model on the second chunk, this time updating the previously computed weights ${\theta_1} \rightarrow {\theta_2}$!
- We rinse and repeat until the end of the dataset

❗️Not all Machine Learning models support incremental learning; only *parametric* models $f_{\theta}$ that are based on *iterative update methods* like Gradient Descent support it
- In **scikit-learn**, `model.partial_fit()` is only available for the SGDRegressor/Classifier and a few others ([read this carefully 📚](https://scikit-learn.org/0.15/modules/scaling_strategies.html#incremental-learning)).
- In **TensorFlow** and other Deep Learning frameworks, training is always iterative, and incremental learning is the default behavior! You just need to avoid calling `model.initialize()` between two chunks!

❗️Do not confuse `chunk_size` with `batch_size` from Deep Learning

👉 For each (big) chunk, your model will read data in many (small) batches over several epochs

<img src='https://wagon-public-datasets.s3.amazonaws.com/data-science-images/07-ML-OPS/train_by_chunk.png'>

👍 **Pros:** this universal approach is framework-independent; you can use it with `scikit-learn`, XGBoost, TensorFlow, etc.

👎 **Cons:** the model will be biased towards fitting the *latest* chunk better than the *first* ones. In our case, it is not a problem as our training dataset is shuffled, but it is important to keep that in mind when we do a partial fit of our model with newer data once it is in production.

<br>

<details>
  <summary markdown='span'><strong>🤔 Do we really need chunks with TensorFlow?</strong></summary>

Granted, thanks to TensorFlow datasets you will not always need "chunks" as you can use batch-by-batch dataset loading as seen below

```python
import tensorflow as tf

ds = tf.data.experimental.make_csv_dataset(data_processed_all.csv, batch_size=256)
model.fit(ds)
```

We will see that in Recap. Still, in this challenge, we would like to teach you the universal method of incrementally fitting in chunks, as it applies to any framework, and will prove useful to *partially retrain* your model with newer data once it is put in production.
</details>

<br>

## 6.2) Your turn - code `def train()`

**Try to code the new route given below by `def train()` in your `ml_logic.interface.main_local` module; copy and paste the code below to get started**

Again, start with a very small dataset size, then finally train your model on 500k rows.

[//]: # (  🚨 Code below is not the single source of truth 🚨 )

<details>
  <summary markdown='span'><strong>👇 Code to copy 👇</strong></summary>

```python
def train(min_date:str = '2009-01-01', max_date:str = '2015-01-01') -> None:
    """
    Incremental train on the (already preprocessed) dataset locally stored.
    - Loading data chunk-by-chunk
    - Updating the weight of the model for each chunk
    - Saving validation metrics at each chunks, and final model weights on local disk
    """

    print(Fore.MAGENTA + "\n ⭐️ Use case:train by batch" + Style.RESET_ALL)
    from taxifare.ml_logic.registry import save_model, save_results
    from taxifare.ml_logic.model import (compile_model, initialize_model, train_model)

    data_processed_path = Path(LOCAL_DATA_PATH).joinpath("processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv")
    model = None
    metrics_val_list = []  # store each val_mae of each chunk

    # Iterate in chunks and partial fit on each chunk
    chunks = pd.read_csv(data_processed_path,
                         chunksize=CHUNK_SIZE,
                         header=None,
                         dtype=DTYPES_PROCESSED)

    for chunk_id, chunk in enumerate(chunks):
        print(f"training on preprocessed chunk n°{chunk_id}")
        # You can adjust training params for each chunk if you want!
        learning_rate = 0.0005
        batch_size = 256
        patience=2
        split_ratio = 0.1 # Higher train/val split ratio when chunks are small! Feel free to adjust.

        # Create (X_train_chunk, y_train_chunk, X_val_chunk, y_val_chunk)
        train_length = int(len(chunk)*(1-split_ratio))
        chunk_train = chunk.iloc[:train_length, :].sample(frac=1).to_numpy()
        chunk_val = chunk.iloc[train_length:, :].sample(frac=1).to_numpy()

        X_train_chunk = chunk_train[:, :-1]
        y_train_chunk = chunk_train[:, -1]
        X_val_chunk = chunk_val[:, :-1]
        y_val_chunk = chunk_val[:, -1]

        # Train a model *incrementally*, and store the val MAE of each chunk in `metrics_val_list`
        # YOUR CODE HERE

    # Return the last value of the validation MAE
    val_mae = metrics_val_list[-1]

    # Save model and training params
    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience,
        incremental=True,
        chunk_size=CHUNK_SIZE
    )

    print(f"✅ Trained with MAE: {round(val_mae, 2)}")

     # Save results & model
    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ train() done")

def pred(X_pred: pd.DataFrame = None) -> np.ndarray:

    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    from taxifare.ml_logic.registry import load_model
    from taxifare.ml_logic.preprocessor import preprocess_features

    if X_pred is None:
       X_pred = pd.DataFrame(dict(
           pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz='UTC')],
           pickup_longitude=[-73.950655],
           pickup_latitude=[40.783282],
           dropoff_longitude=[-73.984365],
           dropoff_latitude=[40.769802],
           passenger_count=[1],
       ))

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")
    return y_pred

```

</details>

**🧪 Test your code**

Check it out with `make test_train_by_chunk`

✅ When you are all green, track your results on kitt with `make test_kitt`

🏁 🏁 🏁 🏁 Congratulations! 🏁 🏁 🏁 🏁


</details>


# 🪐 Enter the Dimension of Cloud Computing! 🚀

In the previous unit, you have **packaged** 📦 the notebook of the _WagonCab_ Data Science team, and updated the code with **chunk-processing** so that the model could be trained on the full _TaxiFare_ dataset despite running "small" local machine.

☁️ In this unit, you will learn how to dispatch work to a pool of **cloud resources** instead of using your local machine.

💪 As you can (in theory) now access machine with the RAM-size of your choice, we'll consider that you don't need any "chunk-by-chunk" logic anymore!

🎯 Today, you will refactor previous unit codebase so as to:
- Fetch all your environment variable from a single `.env` file instead of updating `params.py`
- Load raw data from Le Wagon Big Query all at once on memory (no chunk)
- Cache a local CSV copy to avoid query it twice
- Process data
- Upload processed data on your own Big Query table
- Download processed data (all at once)
- Cache a local CSV copy to avoid query it twice
- Train your model on this processed data
- Store model weights on your own Google Cloud Storage (GCS bucket)

Then, you'll provision a Virtual Machine (VM) so as to run all this workflow on the VM !

Congratulation, you just grow from a **Data Scientist** into an full **ML Engineer**!
You can now sell your big GPU-laptop and buy a lightweight computer like real ML practitioners 😝

---

<br>

## 1️⃣ New taxifare package setup

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>


### Project Structure

👉 From now on, you will start each new challenge with the solution of the previous challenge

👉 Each new challenge will bring in an additional set of features

Here are the main files of interest:
```bash
.
├── .env                            # ⚙️ Single source of all config variables
├── .envrc                          # 🎬 .env automatic loader (used by direnv)
├── Makefile                        # New commands "run_train", "run_process", etc..
├── README.md
├── requirements.txt
├── setup.py
├── taxifare
│   ├── __init__.py
│   ├── interface
│   │   └── main_local.py           # 🚪 (OLD) entry point
│   │   └── main.py                 # 🚪 (NEW) entry point: No more chunks 😇 - Just process(), train()
│   ├── ml_logic
│       ├── data.py                 # (UPDATED) Loading and storing data from/to Big Query !
│       ├── registry.py             # (UPDATED) Loading and storing model weights from/to Cloud Storage!
│       ├── ...
│   ├── params.py                   # Simply load all .env variables into python objects
│   └── utils.py
└── tests
```


#### ⚙️ `.env.sample`

This file is a _template_ designed to help you create a `.env` file for each challenge. The `.env.sample` file contains the variables required by the code and expected in the `.env` file. 🚨 Keep in mind that the `.env` file **should never be tracked with Git** to avoid exposing its content, so we have added it to your `.gitignore`.

#### 🚪 `main.py`

Bye bye `taxifare.interface.main_local` module, you served us well ❤️

Long live `taxifare.interface.main`, our new package entry point ⭐️ to:

- `preprocess`: preprocess the data and store `data_processed`
- `train`: train on processed data and store model weights
- `evaluate`: evaluate the performance of the latest trained model on new data
- `pred`: make a prediction on a `DataFrame` with a specific version of the trained model


🚨 One main change in the code of the package is that we chose to delegate some of its work to dedicated modules in order to limit the size of the `main.py` file. The main changes concern:

- The project configuration: Single source of truth is `.env`
  - `.envrc` tells `direnv` to loads the `.env` as environment variables
  - `params.py` then loads all these variable in python, and should not be changed manually anymore

- `registry.py`: the code evolved to store the trained model either locally or - _spoiler alert_ - in the cloud
  - Notice the new env variable `MODEL_TARGET` (`local` or `gcs`)

- `data.py` has refactored 2 methods that we'll use heavily in `main.py`
  - `get_data_with_cache()` (get some data from Big Query or cached CSV if exists)
  - `load_data_to_bq()` (upload some data to BQ)



### Setup

#### Install `taxifare` version `0.0.7`

**💻 Install the new package version**
```bash
make reinstall_package # always check what make do in Makefile
```

**🧪 Check the package version**
```bash
pip list | grep taxifare
# taxifare               0.0.7
```

#### Setup direnv & .env

Our goal is to be able to configure the behavior of our _package_ 📦 depending on the value of the variables defined in a `.env` project configuration file.

**💻 In order to do so, we will install the `direnv` shell extension.** Its job is to locate the nearest `.env` file in the parent directory structure of the project and load its content into the environment.

``` bash
# MacOS
brew install direnv

# Ubuntu (Linux or Windows WSL2)
sudo apt update
sudo apt install -y direnv
```
Once `direnv` is installed, we need to tell `zsh` to load `direnv` whenever the shell starts

``` bash
code ~/.zshrc
```

The list of plugins is located in the beginning of the file and should look like this when you add `direnv`:

``` bash
plugins=(...direnv)
```

Start a new `zsh` window in order to load `direnv`

**💻 At this point, `direnv` is still not able to load anything, as there is no `.env` file, so let's create one:**

- Duplicate the `env.sample` file and rename the duplicate as `.env`
- Enable the project configuration with `direnv allow .` (the `.` stands for _current directory_)

🧪 Check that `direnv` is able to read the environment variables from the `.env` file:

```bash
echo $DATA_SIZE
# 1k --> Let's keep it small!
```

From now on, every time you need to update the behavior of the project:
1. Edit `.env`, save it
2. Then
```bash
direnv reload . # to reload your env variables 🚨🚨
```

**☝️ You *will* forget that. Prove us wrong 😝**

```bash
# Ok so, for this unit, alway keep data size values small (good practice for dev purposes)
DATA_SIZE=1k
CHUNK_SIZE=200
```

</details>

## 2️⃣ GCP Setup

<details>
<summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

**Google Cloud Platform** will allow you to allocate and use remote resources in the cloud. You can interact with it via:
- 🌐 [console.cloud.google.com](https://console.cloud.google.com)
- 💻 Command Line Tools
  - `gcloud`
  - `bq` (big query - SQL)
  - `gsutils` (cloud storage - buckets)


### a) `gcloud` CLI

- Find the `gcloud` command that lists your own **GCP project ID**.
- 📝 Fill in the `GCP_PROJECT` variable in the `.env` project configuration with the ID of your GCP project
- 🧪 Run the tests with `make test_gcp_project`

<details>
  <summary markdown='span'><strong>💡 Hint </strong></summary>


  You can use the `-h` or the `--help` (more details) flags in order to get contextual help on the `gcloud` commands or sub-commands; use `gcloud billing -h` to get the `gcloud billing` sub-command's help, or `gcloud billing --help` for more detailed help.

  👉 Pressing `q` is usually the way to exit help mode if the command did not terminate itself (`Ctrl + C` also works)

  Also note that running `gcloud` without arguments lists all the available sub-commands by group.

</details>

### b) Cloud Storage (GCS) and the `gsutil` CLI

The second CLI tool that you will use often allows you to deal with files stored within **buckets** on Cloud Storage.

We'll use it to store large & unstructured data such as model weights :)

**💻 Create a bucket in your GCP account using `gsutil`**

- Make sure to create the bucket where you are located yourself (use `GCP_REGION` in the `.env`)
- Fill also the `BUCKET_NAME` variable with the name of your choice (must be globally unique and lower case!)

e.g.
```bash
BUCKET_NAME = taxifare_<user.github_nickname>
```
- `direnv reload .` ;)

Tips: The CLI can interpolate `.env` variables by prefix them with a `$` sign (e.g. `$GCP_REGION`)
<details>
  <summary markdown='span'>🎁 Solution</summary>

```bash
gsutil ls                               # list buckets

gsutil mb \                             # make bucket
    -l $GCP_REGION \
    -p $GCP_PROJECT \
    gs://$BUCKET_NAME                     # make bucket

gsutil rm -r gs://$BUCKET_NAME               # delete bucket
```
You can also use the [Cloud Storage console](https://console.cloud.google.com/storage/) in order create a bucket or list the existing buckets and their content.

Do you see how much slower the GCP console (web interface) is compared to the command line?

</details>

**🧪 Run the tests with `make test_gcp_bucket`**

### c) Big Query and the `bq` CLI

Biq Query is a data-warehouse, used to store structured data, that can be queried rapidly.

💡 To be more precise, Big Query is an online massively-parallel **Analytical Database** (as opposed to **Transactional Database**)

- Data is stored by columns (as opposed to rows on PostGres for instance)
- It's optimized for large transformation such as `group-by`, `join`, `where` etc...easily
- But it's not optimized for frequent row-by-row insert/delete

Le WagonCab is actually using a managed postgreSQL (e.g. [Google Cloud SQL](https://cloud.google.com/sql)) as its main production database on which it's Django app is storing / reading hundred thousands of individual transactions per day!

Every night, Le WagonCab launch a "database replication" job that applies the daily diffs of the "main" postgresSQL into the "replica" Big Query warehouse. Why?
- Because you don't want to run queries directly against your production-database! That could slow down your users.
- Because analysis is faster/cheaper on columnar databases
- Because you also want to integrate other data in your warehouse to JOIN them (e.g marketing data from Google Ads...)

👉 Back to our business:

**💻 Let's create our own dataset where we'll store & query preprocessed data !**

- Using `bq` and the following env variables, create a new _dataset_ called `taxifare` on your own `GCP_PROJECT`

```bash
BQ_DATASET=taxifare
BQ_REGION=...
GCP_PROJECT=...
```

- Then add 3 new _tables_ `processed_1k`, `processed_200k`, `processed_all`

<details>
  <summary markdown='span'>💡 Hints</summary>

Although the `bq` command is part of the **Google Cloud SDK** that you installed on your machine, it does not seem to follow the same help pattern as the `gcloud` and `gsutil` commands.

Try running `bq` without arguments to list the available sub-commands.

What you are looking for is probably in the `mk` (make) section.
</details>

<details>
  <summary markdown='span'><strong>🎁 Solution </strong></summary>

``` bash
bq mk \
    --project_id $GCP_PROJECT \
    --data_location $BQ_REGION \
    $BQ_DATASET

bq mk --location=$GCP_REGION $BQ_DATASET.processed_1k
bq mk --location=$GCP_REGION $BQ_DATASET.processed_200k
bq mk --location=$GCP_REGION $BQ_DATASET.processed_all

bq show
bq show $BQ_DATASET
bq show $BQ_DATASET.processed_1k

```

</details>

**🧪 Run the tests with `make test_big_query`**


🎁 Look at `make reset_all_files` directive --> It resets all local files (csvs, models, ...) and data from bq tables and buckets, but preserve local folder structure, bq tables schema, and gsutil buckets.

Very useful to reset state of your challenge if you are uncertain and you want to debug yourself!

👉 Run `make reset_all_files` safely now, it will remove files from unit 01 and make it clearer

👉 Run `make show_sources_all` to see that you're back from a blank state!

✅ When you are all set, track your results on Kitt with `make test_kitt` (don't wait, this takes > 1min)

</details>

## 3️⃣ ⚙️ Train locally, with data on the cloud !

<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>

🎯 Your goal is to fill-up `taxifare.interface.main` so that you can run every 4 routes _one by one_

```python
if __name__ == '__main__':
    # preprocess()
    # train()
    # evaluate()
    # pred()
```

To do so, you can either:

- 🥵 Uncomment the routes above, one after the other, and run `python -m taxifare.interface.main` from your Terminal

- 😇 Smarter: use each of the following `make` commands that we created for you below

💡 Make sure to read each function docstring carefully
💡 Don't try to parallelize route completion. Fix them one after the other.
💡 Take time to read carefully the tracebacks, and add breakpoint() to your code or to the test itself (you are 'engineers' now)!

**Preprocess**

💡 Feel free to refer back to `main_local.py` when needed! Some of the syntax can be re-used

```bash
# Call your preprocess()
make run_preprocess
# Then test this route, but with all combinations of states (.env, cached_csv or not)
make test_preprocess
```

**Train**

💡 Be sure to understand what happens when MODEL_TARGET = 'gcs' vs 'local'
💡 We advise you to set `verbose=0` on model training to shorter your logs!

```bash
make run_train
make test_train
```

**Evaluate**

Be sure to understand what happens when MODEL_TARGET = 'gcs' vs 'local'
```bash
make run_evaluate
make test_evaluate
```

**Pred**

This one is easy
```bash
make run_pred
make test_pred
```

✅ When you are all set, track your results on Kitt with `make test_kitt`

🏁 Congrats for the heavy refactoring! You now have a very robust package that can be deployed in the cloud to be used with `DATA_SIZE='all'` 💪

</details>

## 4️⃣ Train in the Cloud with Virtual Machines


<details>
  <summary markdown='span'><strong>❓Instructions (expand me)</strong></summary>


### Enable the Compute Engine Service

In GCP, many services are not enabled by default. The service to activate in order to use _virtual machines_ is **Compute Engine**.

**❓How do you enable a GCP service?**

Find the `gcloud` command to enable a **service**.

<details>
  <summary markdown='span'>💡 Hints</summary>

[Enabling an API](https://cloud.google.com/endpoints/docs/openapi/enable-api#gcloud)
</details>

### Create your First Virtual Machine

The `taxifare` package is ready to train on a machine in the cloud. Let's create our first *Virtual Machine* instance!

**❓Create a Virtual Machine**

Head over to the GCP console, specifically the [Compute Engine page](https://console.cloud.google.com/compute). The console will allow you to easily explore the available options. Make sure to create an **Ubuntu** instance (read the _how-to_ below and have a look at the _hint_ after it).

<details>
  <summary markdown='span'><strong> 🗺 How to configure your VM instance </strong></summary>


  Let's explore the options available. The top right of the interface gives you a monthly estimate of the cost for the selected parameters if the VM remains online all the time.

  The default options should be enough for what we want to do now, except for one: we want to choose the operating system that the VM instance will be running.

  Go to the **"Boot disk"** section, click on **"CHANGE"** at the bottom, change the **operating system** to **Ubuntu**, and select the latest **Ubuntu xx.xx LTS x86/64** (Long Term Support) version.

  Ubuntu is the [Linux distro](https://en.wikipedia.org/wiki/Linux_distribution) that will resemble the configuration on your machine the most, following the [Le Wagon setup](https://github.com/lewagon/data-setup). Whether you are on a Mac, using Windows WSL2 or on native Linux, selecting this option will allow you to play with a remote machine using the commands you are already familiar with.
</details>

<details>
  <summary markdown='span'><strong>💡 Hint </strong></summary>

  In the future, when you know exactly what type of VM you want to create, you will be able to use the `gcloud compute instances` command if you want to do everything from the command line; for example:

  ``` bash
  INSTANCE=taxi-instance
  IMAGE_PROJECT=ubuntu-os-cloud
  IMAGE_FAMILY=ubuntu-2204-lts

  gcloud compute instances create $INSTANCE --image-project=$IMAGE_PROJECT --image-family=$IMAGE_FAMILY
  ```
</details>

**💻 Fill in the `INSTANCE` variable in the `.env` project configuration**


### Setup your VM

You have access to virtually unlimited computing power at your fingertips, ready to help with trainings or any other task you might think of.

**❓How do you connect to the VM?**

The GCP console allows you to connect to the VM instance through a web interface:

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-vm-ssh.png"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-vm-ssh.png" height="450" alt="gce vm ssh"></a><a href="https://wagon-public-datasets.s3.amazonaws.com/07-ML-Ops/02-Cloud-Training/GCE_SSH_in_browser.png"><img style="margin-left: 15px;" src="https://wagon-public-datasets.s3.amazonaws.com/07-ML-Ops/02-Cloud-Training/GCE_SSH_in_browser.png" height="450" alt="gce console ssh"></a>

You can disconnect by typing `exit` or closing the window.

A nice alternative is to connect to the virtual machine right from your command line 🤩

<a href="https://wagon-public-datasets.s3.amazonaws.com/07-ML-Ops/02-Cloud-Training/GCE_SSH_in_terminal.png"><img src="https://wagon-public-datasets.s3.amazonaws.com/07-ML-Ops/02-Cloud-Training/GCE_SSH_in_terminal.png" height="450" alt="gce ssh"></a>

All you need to do is to `gcloud compute ssh` on a running instance and to run `exit` when you want to disconnect 🎉

``` bash
INSTANCE=taxi-instance

gcloud compute ssh $INSTANCE
```

<details>
  <summary markdown='span'><strong>💡 Error 22 </strong></summary>


  If you encounter a `port 22: Connection refused` error, just wait a little more for the VM instance to complete its startup.

  Just run `pwd` or `hostname` if you ever wonder on which machine you are running your commands.
</details>

**❓How do you setup the VM to run your python code?**

Let's run a light version of the [Le Wagon setup](https://github.com/lewagon/data-setup).

**💻 Connect to your VM instance and run the commands of the following sections**

<details>
  <summary markdown='span'><strong> ⚙️ <code>zsh</code> and <code>omz</code> (expand me)</strong></summary>

The **zsh** shell and its **Oh My Zsh** framework are the _CLI_ configuration you are already familiar with. When prompted, make sure to accept making `zsh` the default shell.

``` bash
sudo apt update
sudo apt install -y zsh
sh -c "$(curl -fsSL https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

👉 Now the _CLI_ of the remote machine starts to look a little more like the _CLI_ of your local machine
</details>

<details>
  <summary markdown='span'><strong> ⚙️ <code>pyenv</code> and <code>pyenv-virtualenv</code> (expand me)</strong></summary>

Clone the `pyenv` and `pyenv-virtualenv` repos on the VM:

``` bash
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
git clone https://github.com/pyenv/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv
```

Open ~/.zshrc in a Terminal code editor:

``` bash
nano ~/.zshrc
```

Add `pyenv`, `ssh-agent` and `direnv` to the list of `zsh` plugins on the line with `plugins=(git)` in `~/.zshrc`: in the end, you should have `plugins=(git pyenv ssh-agent direnv)`. Then, exit and save (`Ctrl + X`, `Y`, `Enter`).

Make sure that the modifications were indeed saved:

``` bash
cat ~/.zshrc | grep "plugins="
```

Add the pyenv initialization script to your `~/.zprofile`:

``` bash
cat << EOF >> ~/.zprofile
export PYENV_ROOT="\$HOME/.pyenv"
export PATH="\$PYENV_ROOT/bin:\$PATH"
eval "\$(pyenv init --path)"
EOF
```

👉 Now we are ready to install Python

</details>

<details>
  <summary markdown='span'><strong> ⚙️ <code>Python</code> (expand me)</strong></summary>

Add dependencies required to build Python:

``` bash
sudo apt-get update; sudo apt-get install make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev \
python3-dev
```

ℹ️ If a window pops up to ask you which services to restart, just press *Enter*:

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-apt-services-restart.png"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-apt-services-restart.png" width="450" alt="gce apt services restart"></a>

Now we need to start a new user session so that the updates in `~/.zshrc` and `~/.zprofile` are taken into account. Run the command below 👇:

``` bash
zsh --login
```

Install the same python version that you use for the bootcamp, and create a `lewagon` virtual env. This can take a while and look like it is stuck, but it is not:

``` bash
# e.g. with 3.10.6
pyenv install 3.10.6
pyenv global 3.10.6
pyenv virtualenv 3.10.6 taxifare-env
pyenv global taxifare-env
```

</details>

<details>
  <summary markdown='span'><strong> ⚙️ <code>git</code> authentication with GitHub (expand me)</strong></summary>

Copy your private key 🔑 to the _VM_ in order to allow it to access your GitHub account.

⚠️ Run this single command on your machine, not in the VM ⚠️

``` bash
INSTANCE=taxi-instance

# scp stands for secure copy (cp)
gcloud compute scp ~/.ssh/id_ed25519 $USER@$INSTANCE:~/.ssh/
```

⚠️ Then, resume running commands in the VM ⚠️

Register the key you just copied after starting `ssh-agent`:

``` bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

Enter your *passphrase* if asked to.

👉 You are now able to interact with your **GitHub** account from the _virtual machine_
</details>

<details>
  <summary markdown='span'><strong> ⚙️ <em>Python</em> code authentication to GCP (expand me)</strong></summary>

The code of your package needs to be able to access your Big Query data warehouse.

To do so, we will login to your account using the command below 👇

``` bash
gcloud auth application-default login
```

❗️ Note: In a full production environment we would create a service account applying the least privilege principle for the vm but this is the easiest approach for development.

Let's verify that your Python code can now access your GCP resources. First, install some packages:

``` bash
pip install -U pip
pip install google-cloud-storage
```

Then, [run Python code from the _CLI_](https://stackoverflow.com/questions/3987041/run-function-from-the-command-line). This should list your GCP buckets:

``` bash
python -c "from google.cloud import storage; \
    buckets = storage.Client().list_buckets(); \
    [print(b.name) for b in buckets]"
```

</details>

Your _VM_ is now fully operational with:
- A python venv (lewgon) to run your code
- The credentials to connect to your _GitHub_ account
- The credentials to connect to your _GCP_ account

The only thing that is missing is the code of your project!

**🧪 Let's run a few tests inside your _VM Terminal_ before we install it:**

- Default shell is `/usr/bin/zsh`
    ```bash
    echo $SHELL
    ```
- Python version is `3.10.6`
    ```bash
    python --version
    ```
- Active GCP project is the same as `$GCP_PROJECT` in your `.env` file
    ```bash
    gcloud config list project
    ```

Your VM is now a data science beast 🔥

### Train in the Cloud

Let's run your first training in the cloud!

**❓How do you setup and run your project on the virtual machine?**

**💻 Clone your package, install its requirements**

<details>
  <summary markdown='span'><strong>💡 Hint </strong></summary>

You can copy your code to the VM by cloning your GitHub project with this syntax:

```bash
git clone git@github.com:<user.github_nickname>/{{local_path_to("07-ML-Ops/02-Cloud-training/01-Cloud-training")}}
```

Enter the directory of today's taxifare package (adapt the command):

``` bash
cd <path/to/the/package/model/dir>
```

Create directories to save the model and its parameters/metrics:

``` bash
make reset_local_files
```

Create a `.env` file with all required parameters to use your package:

``` bash
cp .env.sample .env
```

Fill in the content of the `.env` file (complete the missing values, change any values that are specific to your virtual machine):

``` bash
nano .env
```

Install `direnv` to load your `.env`:

``` bash
sudo apt update
sudo apt install -y direnv
```

ℹ️ If a window pops up to ask you which services to restart, just press *Enter*.

Reconnect (simulate a user reboot) so that `direnv` works:

``` bash
zsh --login
```

Allow your `.envrc`:

``` bash
direnv allow .
```

Install the taxifare package (and all its dependencies)!

``` bash
pip install .
```

</details>

**🔥 Run the preprocessing and the training in the cloud 🔥**!

``` bash
make run_all
```

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-train-ssh.png"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-train-ssh.png" height="450" alt="gce train ssh"></a>

> `Project not set` error from GCP services? You can add a `GCLOUD_PROJECT` environment variable that should be the same as your `GCP_PROJECT`

🧪 Track your progress on Kitt to conclude (from your VM)

```bash
make test_kitt
```

**🏋🏽‍♂️ Go Big: re-run everything with `DATA_SIZE = 'all'`  `CHUNK_SIZE=100k` chunks for instance 🏋🏽‍♂️**!

**🏁 Switch OFF your VM to finish 🌒**

You can easily start and stop a VM instance from the GCP console, which allows you to see which instances are running.

<a href="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-vm-start.png"><img src="https://wagon-public-datasets.s3.amazonaws.com/data-science-images/DE/gce-vm-start.png" height="450" alt="gce vm start"></a>

<details>
  <summary markdown='span'><strong>💡 Hint </strong></summary>

A faster way to start and stop your virtual machine is to use the command line. The commands still take some time to complete, but you do not have to navigate through the GCP console interface.

Have a look at the `gcloud compute instances` command in order to start, stop, or list your instances:

``` bash
INSTANCE=taxi-instance

gcloud compute instances stop $INSTANCE
gcloud compute instances list
gcloud compute instances start $INSTANCE
```
</details>

🚨 Computing power does not grow on trees 🌳, do not forget to switch the VM **off** whenever you stop using it! 💸

</details>

<br>


🏁 Remember: Switch OFF your VM with `gcloud compute instances stop $INSTANCE`



