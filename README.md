# misc-exercises-ml

## General Useful Libraries for AI:
- numpy (Linear Algebra)
- pandas (Data Processing, CSV file I/O)
- matplotlib (Visualization with Python)
- seaborn (Statistical Data Visualization)

## AI-Specific Libraries:
- scikit-learn (Machine Learning with Python)

## General step-by-step:
- Load df and get X matrix and y vector
- Preprocess data (encode class labels to enums, scale samples...)
- Split train/test
- Train the model
- Rate the results (score, confusion mtx...)

## Common questions for a good data...
- Data augmentation?
- Balanced dataset?
- Are labels encoded?
- Are samples scaled?

## PCA
- Reduz a redundância de colunas que estão correlacionadas (ToDo: translate strictly)
- Should be applied after data splitting and before model training
- Must be trained only upon XTrain and just applied the same transform to XTest (fitting upon XTest is basically cheating, because it's supposedly real and unknown data, thus, shouldn't be used to train PCA)
