Face Recognition Using PCA and SVM

This project implements a face recognition system using Principal Component Analysis (PCA) and a non-linear Support Vector Machine (SVM). The goal is to classify individuals based on facial images by first reducing the dimensionality of the image data and then training a classifier.

The project uses the AT&T Face Dataset, which contains grayscale face images of multiple individuals.

Dataset

The dataset comes from the AT&T Laboratories Cambridge Face Database.

Dataset characteristics

• 400 total images
• 40 individuals
• 10 images per person
• Image size: 64 × 64 pixels
• 4096 features per image when flattened

Each image is labeled with the person ID in label.csv.

Dataset folder structure used in this project:

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
Methodology

The face recognition pipeline consists of the following steps.

1. Feature Extraction

Each image is treated as a data sample.

Since images are 64 × 64 pixels, each image is converted into a 4096 dimensional feature vector by flattening the pixel matrix.

64 × 64 = 4096 features
2. Data Normalization

Before dimensionality reduction, each feature column is normalized using:

sklearn.preprocessing.scale

This standardizes each pixel feature so that:

mean = 0
standard deviation = 1

Normalization is important because PCA and SVM are sensitive to feature scale.

3. Train/Test Split

The dataset is split into training and testing sets using:

test_size = 0.25
random_state = 5

Resulting dataset sizes:

Dataset	Samples
Training	300
Testing	100
4. Dimensionality Reduction using PCA

The original feature dimension is 4096, which is very high.

Principal Component Analysis (PCA) is used to reduce the dimensionality to:

k = 50 principal components

Important implementation detail:

PCA is fit only on the training set, then applied to both training and testing data.

X_Train_new = my_pca.fit_transform(X_Train)
X_Test_new = my_pca.transform(X_Test)

After PCA:

Dataset	Features
Training	50
Testing	50
5. Face Classification using SVM

A non-linear Support Vector Machine (SVM) with an RBF kernel is used for classification.

Model parameters:

SVC(
    C=1,
    kernel='rbf',
    gamma=0.0005,
    random_state=1
)

The trained SVM predicts the identity of individuals in the testing dataset.

Performance is evaluated using:

• Accuracy
• Confusion Matrix

6. Hyperparameter Optimization

To improve performance, GridSearchCV is used to find the best value of the regularization parameter C.

Search space:

C = [0.1, 1, 10, 100, 1e3, 5e3, 1e4, 5e4, 1e5]

The grid search uses:

10-fold cross-validation
scoring = 'accuracy'

Before GridSearchCV, PCA is performed on the entire normalized dataset, as specified in the assignment instructions.

Technologies Used

Python
NumPy
Pandas
Matplotlib
Scikit-learn

Main algorithms:

• Principal Component Analysis (PCA)
• Support Vector Machine (SVM)
• GridSearchCV

How to Run the Project

Clone the repository

git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

Navigate to the project directory

cd YOUR_REPO_NAME

Install required packages

pip install numpy pandas matplotlib scikit-learn

Open the Jupyter notebook

jupyter notebook

Run

Face_Recognition.ipynb
Example Output

The notebook outputs:

• Test accuracy of the SVM classifier
• Confusion matrix for the 40-class classification problem
• Best value of C found using GridSearchCV
