# Face Recognition Using PCA and SVM

This project implements a face recognition system using **Principal Component Analysis (PCA)** and a **non-linear Support Vector Machine (SVM)**. The goal is to classify individuals based on facial images by first reducing the dimensionality of the image data and then training a classifier.

The project uses the **AT&T Face Dataset**, which contains grayscale face images of multiple individuals.

---

## Dataset

The dataset comes from the [AT&T Laboratories Cambridge Face Database](https://git-disl.github.io/GTDLBench/datasets/att_face_dataset/).

**Dataset characteristics:**

| Property | Value |
|---|---|
| Total images | 400 |
| Individuals | 40 |
| Images per person | 10 |
| Image size | 64 × 64 pixels |
| Features per image (flattened) | 4,096 |

Each image is labeled with the person ID in `label.csv`.

**Folder structure:**

```
Face_Project/
│
├── Face/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   ├── 399.jpg
│   └── label.csv
│
└── Face_Recognition.ipynb
```

---

## Methodology

### 1. Feature Extraction

Each 64 × 64 image is flattened into a **4,096-dimensional feature vector**.

### 2. Data Normalization

Each feature column is standardized using `sklearn.preprocessing.scale` so that every pixel feature has **mean = 0** and **standard deviation = 1**. This is important because both PCA and SVM are sensitive to feature scale.

### 3. Train/Test Split

The dataset is split with `test_size=0.25` and `random_state=5`:

| Set | Samples |
|---|---|
| Training | 300 |
| Testing | 100 |

### 4. Dimensionality Reduction (PCA)

PCA reduces the feature space from 4,096 down to **50 principal components**.

> **Important:** PCA is fit only on the training set, then applied to both sets.

```python
X_Train_new = my_pca.fit_transform(X_Train)
X_Test_new  = my_pca.transform(X_Test)
```

### 5. Classification (SVM)

A non-linear SVM with an **RBF kernel** is used for classification:

```python
SVC(C=1, kernel='rbf', gamma=0.0005, random_state=1)
```

Performance is evaluated using **accuracy** and a **confusion matrix**.

### 6. Hyperparameter Optimization

`GridSearchCV` is used to find the best regularization parameter `C`:

```python
C = [0.1, 1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5]
```

- **Cross-validation:** 10-fold
- **Scoring metric:** accuracy

> **Note:** Before GridSearchCV, PCA is performed on the entire normalized dataset, as specified in the assignment instructions.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn (PCA, SVM, GridSearchCV)

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```

2. **Navigate to the project directory**
   ```bash
   cd YOUR_REPO_NAME
   ```

3. **Install required packages**
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

4. **Open and run the notebook**
   ```bash
   jupyter notebook Face_Recognition.ipynb
   ```

---

## Example Output

The notebook outputs:

- Test accuracy of the SVM classifier
- Confusion matrix for the 40-class classification problem
- Best value of `C` found using GridSearchCV
