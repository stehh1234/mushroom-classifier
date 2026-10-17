# Basic ML: is this mushroom poisonous?

The same recipe as Plato (build a model with plain **scikit-learn**, save it with **joblib**,
serve it with **FastAPI**), applied to a new dataset: 8,124 mushrooms from a field guide, each
**edible** or **poisonous**, described by 22 features such as smell, cap colour and spore print.

> A teaching project, not a foraging tool. Never eat a wild mushroom because a model said so.

What is different from churn, and worth discussing in class:

- **Every feature is a category, stored as one letter.** `app/codes.py` translates the letters
  into words, and `OneHotEncoder` turns the words into 0/1 columns for the model.
- **The mistakes are not equal.** Calling a poisonous mushroom edible is far worse than a false
  alarm, so the threshold is chosen to catch *every* poisonous mushroom first.
- **The data is easy, and that is the lesson.** Smell alone is 98.5% accurate and the full model
  is perfect. Perfect scores call for suspicion: look for leaks, then ask whether real inputs
  (a beginner's description, a photo) would be this clean.

```
Mashroom/
├── data/
│   ├── mushrooms.csv                 8,124 mushrooms, `class` = e (edible) or p (poisonous)
│   └── README.md                     where it comes from, what the columns hold
├── notebooks/
│   └── 1_mushroom_model.ipynb        build the model step by step, with plots
├── train_mushroom.py                 the same as the notebook, in one script
├── models/                           mushroom.joblib (created by training)
├── app/
│   ├── codes.py                      one-letter codes -> words, for every feature
│   ├── ml.py                         load the model, make predictions (no FastAPI here)
│   ├── schemas.py                    what the API accepts and returns (Pydantic)
│   └── main.py                       the FastAPI app: routes, loading the model at startup
├── tests/                            pytest: training quality + every endpoint
└── requests.http                     ready-made requests (VS Code REST Client, JetBrains)
```

## Setup

Python 3.11 or newer.

```bash
cd Mashroom
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Windows, if this folder sits deep inside OneDrive, put the environment at a short path
instead (`python -m venv C:\venvs\mashroom`): JupyterLab has files whose full path goes over
Windows' 260-character limit.

## Part 1: build the model

```bash
jupyter lab notebooks             # step by step, with explanations and plots
python train_mushroom.py          # ~15 s, prints every step
```

Both write `models/mushroom.joblib`: the fitted model **and** what the API needs to use it (the
feature order, the decision threshold, the scores, the training date, the scikit-learn version).

| Step | What it teaches |
|---|---|
| 1. Load and translate | `p` means *pungent*, *pink* or *pendant* depending on the column: decode per column. `veil_type` has one value, so it is dropped; `stalk_root` is unknown for 2,480 rows, so `missing` becomes its own value. |
| 2. Train / test split | Judge a model on data it has never seen; `stratify` keeps the share of poisonous equal. |
| 3. Baseline | "Always edible" is 51.8% accurate and catches 0 poisonous mushrooms. Watch **recall on poisonous**. |
| 4. One feature at a time | Odor alone: 98.5%. Spore print colour: 86.9%. A few features carry most of the signal. |
| 5. Cross-validation | Logistic regression and a random forest both reach ROC AUC 1.000. Be suspicious of perfect scores. |
| 6. Safe threshold | Keep the thresholds that flag every poisonous mushroom in cross-validation; pick the one with the fewest false alarms, then the lowest (0.09). |
| 7. Test once | 783 of 783 poisonous caught, 0 of 842 edible flagged by mistake. |
| 8. Without the nose | Drop `odor` and it is still 100%: the other features are redundant with it. That is why permutation importance shows almost nothing here. |
| 9. Save | `joblib.dump` a bundle, not only the model. |

The notebook adds a "what if" (step 10): the edible mushroom from the second row of the file,
but smelling foul. Its poisonous probability only rises to about 0.31, and it is flagged only
because the threshold is a cautious 0.09. With the default 0.5 it would be called edible.

## Part 2: serve it with FastAPI

```bash
fastapi dev app/main.py           # http://127.0.0.1:8000, reloads when the code changes
```

Open <http://127.0.0.1:8000/docs>: every endpoint with an example you can send ("Try it out"),
and a drop-down of the allowed words for every feature.

| Method | Path | Does |
|---|---|---|
| GET | `/health` | Is the API up, is the model loaded |
| GET | `/models` | How the model was trained and how well it scored |
| POST | `/mushroom/predict` | One mushroom → poisonous probability, verdict, threshold |
| POST | `/mushroom/predict/batch` | A list of mushrooms in one call |

```bash
curl -s http://127.0.0.1:8000/mushroom/predict -H 'Content-Type: application/json' -d '{
  "cap_shape": "convex", "cap_surface": "smooth", "cap_color": "brown", "bruises": "yes",
  "odor": "pungent", "gill_attachment": "free", "gill_spacing": "close", "gill_size": "narrow",
  "gill_color": "black", "stalk_shape": "enlarging", "stalk_root": "equal",
  "stalk_surface_above_ring": "smooth", "stalk_surface_below_ring": "smooth",
  "stalk_color_above_ring": "white", "stalk_color_below_ring": "white", "veil_color": "white",
  "ring_number": "one", "ring_type": "pendant", "spore_print_color": "black",
  "population": "scattered", "habitat": "urban"
}'
# {"poisonous_probability":1.0,"verdict":"poisonous","threshold":0.09}
```

More requests, including a "what if" and invalid ones, are in [`requests.http`](requests.http).

### How the app is built

Read the files in this order:

1. [`app/codes.py`](app/codes.py): one dictionary per feature, letter → word. Training uses it
   to decode the CSV; the API uses it to know which words are allowed.
2. [`app/ml.py`](app/ml.py): plain Python, no web code. `load()` reads a `.joblib` file,
   `predict_mushroom()` turns a list of mushrooms into a list of answers.
3. [`app/schemas.py`](app/schemas.py): every feature is a `Literal` of its allowed words, built
   from `codes.py`. A misspelt smell (`"lemon"`), a missing field or an extra one (`veil_type`)
   is a `422` with a clear message, before the model ever sees it.
4. [`app/main.py`](app/main.py): the routes. As in Plato: load the model **once** in the
   `lifespan`, keep routes thin, answer `503` with the command to run if the model is missing,
   and score batches in one model call.

After retraining, restart the API: the model file is read at startup.

## Part 3: test it

```bash
pytest
```

The tests train the model into a temporary folder once (so they never touch `models/`), check
that every letter in the CSV is translated and that the model is accurate and safe, then call
every endpoint: a poisonous and an edible mushroom, a batch, invalid input (`422`) and a missing
model (`503`).

## Exercises

On the model (more at the end of the notebook):

1. Train on what you can **see** only: drop `odor`. Then drop `spore_print_color` too (it needs
   a lab-like step). When does the model start missing poisonous mushrooms?
2. Find the smallest set of features that still gives a perfect cross-validated score.
3. Use a `DecisionTreeClassifier(max_depth=4)` and print its rules with `export_text`. Could a
   person follow them in a forest?

On the API:

4. Add `GET /features`: every feature with its allowed words (from `app/codes.py`), for a form.
5. Add a `warning` field to every answer, and a stronger one when the probability is between the
   threshold and 0.5 (the "only cautious because of the threshold" zone).
6. Log every prediction to a CSV file (input, output, time).
7. Write a `Dockerfile` that trains the model at build time and runs `fastapi run`.

## Good to know

- **Only load model files you trust.** joblib (like pickle) can run code when it loads a file.
- **Train and serve with the same scikit-learn version.** The bundle records the version and the
  API warns when it differs.
- **The feature order is part of the model.** The bundle stores it, and the API builds its
  DataFrame with exactly those columns.
- `models/*.joblib` is not committed: anyone can recreate it with `python train_mushroom.py`.
