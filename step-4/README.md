# Step 4 — Majority-class baseline

This is the simplest classification baseline from AI Lab Session 5: predict the
most frequent target class for every test row.

## Dataset compatibility

The script supports both Nim dataset schemas used in this project:

- **Session 5 position dataset:** target column `loses` (the player whose turn it
  is will lose).
- **Step 3 simulated-games dataset:** target column `first_wins` (whether the
  first player wins under optimal play).

If `--target` is omitted, the script selects `loses` first and then
`first_wins`.

## Run

From the repository root, after generating `nim_games.csv` by running the
Step 3 notebook:

```bash
python step-4/baseline.py
```

For another CSV or an explicit target:

```bash
python step-4/baseline.py path/to/data.csv --target loses
```

Dependencies: `pandas` and `scikit-learn`.

## Method and metrics

1. Read the CSV and remove rows with missing target labels.
2. Split the data into 70% train and 30% test using a fixed random seed and
   stratification.
3. Fit `DummyClassifier(strategy="most_frequent")` on the training labels only.
4. Predict the same majority class for every test example.
5. Report test accuracy and recall separately for every target class.

The majority class is selected from the training split, not from the test set.
This avoids using test labels to define the baseline. A high accuracy can still
hide a recall of zero for a minority class; therefore, inspect every class's
recall, not accuracy alone.
