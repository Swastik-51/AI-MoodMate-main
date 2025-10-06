# Python Programming Basics - Learning Repository

This repository contains a comprehensive collection of Python programming examples covering four major areas: Computer Vision, Data Visualization, Natural Language Processing, and NumPy/Machine Learning fundamentals.

## 📁 Repository Structure

```
├── computer_vision/     # OpenCV and image processing examples
├── matplotlib/          # Data visualization and plotting examples
├── NLP/                # Natural Language Processing techniques
└── numpy_library/      # NumPy operations and machine learning basics
```

## 🔗 Table of Contents

- [Computer Vision Examples](#computer-vision-examples)
- [Matplotlib Visualization](#matplotlib-visualization)
- [Natural Language Processing](#natural-language-processing)
- [NumPy and Machine Learning](#numpy-and-machine-learning)
- [Getting Started](#getting-started)

---

## 🎥 Computer Vision Examples

This folder contains 16 OpenCV examples demonstrating fundamental computer vision techniques.

### Core Image Operations

#### [copy_1.py](./computer_vision/copy_1.py) - Camera Stream Capture
**Purpose**: Real-time webcam video capture and display
- **Input**: USB webcam (index 0) or IP camera stream
- **Output**: Live video window display
- **Key Functions**: `cv2.VideoCapture()`, `cv2.imshow()`
- **Exit**: Press 'q' to quit

#### [copy_2.py](./computer_vision/copy_2.py) - Frame Extraction
**Purpose**: Capture and save individual frames from video stream
- **Input**: Webcam feed
- **Output**: Sequential frame images saved to `frames/` directory
- **Features**: Automatic directory creation, frame counting
- **File Format**: JPG images with 6-digit numbering

#### [copy_3.py](./computer_vision/copy_3.py) - Basic Image Loading
**Purpose**: Load and display static images
- **Input**: Sample images from `resources/man.jpg` and `resources/mountainous.jpg`
- **Output**: Image displayed in window
- **Error Handling**: Checks if image loads successfully

### Image Transformations

#### [copy_4.py](./computer_vision/copy_4.py) - Image Flipping
**Purpose**: Demonstrate different image flip operations
- **Input**: Static image file
- **Output**: Four windows showing original and flipped versions
- **Flip Types**: 
  - Vertical flip (0)
  - Horizontal flip (1) 
  - Both axes flip (-1)

#### [copy_5.py](./computer_vision/copy_5.py) - Image Resizing
**Purpose**: Resize images to specific dimensions
- **Input**: Original image
- **Output**: Resized image (300x300 pixels)
- **Features**: 
  - Side-by-side comparison display
  - Saves resized output as `resized_output.jpg`

#### [copy_6.py](./computer_vision/copy_6.py) - Grayscale Conversion
**Purpose**: Convert color images to grayscale
- **Input**: Color image (BGR format)
- **Output**: Grayscale image
- **Color Space**: BGR to GRAY conversion using `cv2.cvtColor()`
- **Save Option**: Outputs `grayscale_output.jpg`

### Image Filtering and Effects

#### [copy_7.py](./computer_vision/copy_7.py) - Gaussian Blur
**Purpose**: Apply blur effect to reduce image noise
- **Input**: Sharp image
- **Output**: Blurred image using 15x15 Gaussian kernel
- **Parameters**: Kernel size (15,15), sigma value 0 (auto-calculated)
- **Use Cases**: Noise reduction, preprocessing for edge detection

#### [copy_8.py](./computer_vision/copy_8.py) - Drawing Shapes and Text
**Purpose**: Create graphics programmatically
- **Canvas**: 500x500 black image created with `np.zeros()`
- **Elements Drawn**:
  - Blue diagonal line from (0,0) to (500,500)
  - Green rectangle from (50,50) to (200,200)
  - Red filled circle at (300,300) with radius 80
  - White text "OpenCV Demo" at bottom

### Image Analysis Techniques

#### [copy_9.py](./computer_vision/copy_9.py) - Binary Thresholding
**Purpose**: Convert grayscale to binary (black/white) image
- **Input**: Grayscale image
- **Threshold**: 127 (pixels above become 255, below become 0)
- **Type**: `THRESH_BINARY` - creates stark black and white contrast
- **Applications**: Object segmentation, preprocessing for contour detection

#### [copy_10.py](./computer_vision/copy_10.py) - Edge Detection
**Purpose**: Detect edges using Canny algorithm
- **Input**: Grayscale image
- **Parameters**: Lower threshold 100, upper threshold 200
- **Output**: Binary edge map highlighting object boundaries
- **Algorithm**: Canny edge detector - industry standard for edge detection

### Advanced Computer Vision

#### [copy_11.py](./computer_vision/copy_11.py) - Face Detection
**Purpose**: Detect human faces using Haar Cascades
- **Algorithm**: Pre-trained Haar Cascade classifier
- **Input**: Color image converted to grayscale for processing
- **Output**: Original image with blue rectangles around detected faces
- **Parameters**: Scale factor 1.1, minimum neighbors 4
- **Applications**: Security systems, photo tagging, biometrics

#### [copy_12.py](./computer_vision/copy_12.py) - Contour Detection
**Purpose**: Find and highlight object boundaries
- **Process**: 
  1. Convert to grayscale
  2. Apply binary threshold
  3. Find contours using `cv2.findContours()`
  4. Draw all contours in green
- **Applications**: Object counting, shape analysis, quality inspection

#### [copy_13.py](./computer_vision/copy_13.py) - Color-Based Object Tracking
**Purpose**: Isolate objects based on color in HSV space
- **Color Target**: Blue objects (HSV range: 100-140 hue, 150-255 saturation)
- **Process**:
  1. Convert BGR to HSV color space
  2. Create color mask using `cv2.inRange()`
  3. Apply mask to original image
- **Applications**: Object tracking, quality control, robotics

#### [copy_14.py](./computer_vision/copy_14.py) - GrabCut Foreground Extraction
**Purpose**: Advanced foreground/background segmentation
- **Algorithm**: GrabCut - interactive segmentation
- **Input**: Region of Interest (ROI) rectangle (50,50,400,500)
- **Process**: 5 iterations of GrabCut algorithm
- **Output**: Extracted foreground object with background removed
- **Applications**: Photo editing, object extraction, image compositing

#### [copy_15.py](./computer_vision/copy_15.py) - Real-Time Color Tracking
**Purpose**: Track blue objects in live video stream
- **Input**: Live webcam feed
- **Processing**: Same blue color detection as copy_13.py but applied to video
- **Display**: Three windows - original frame, color mask, tracked result
- **Applications**: Real-time object tracking, gesture recognition

#### [copy_16.py](./computer_vision/copy_16.py) - Morphological Operations
**Purpose**: Shape-based image processing operations
- **Operations**:
  - **Erosion**: Shrinks white regions (removes noise)
  - **Dilation**: Expands white regions (fills gaps)
- **Kernel**: 5x5 structuring element
- **Applications**: Noise removal, object separation, shape analysis

---

## 📊 Matplotlib Visualization

This folder contains 10 examples demonstrating various plotting and visualization techniques using Matplotlib.

### Basic Plotting

#### [plot1.py](./matplotlib/plot1.py) - Simple Line Plot
**Purpose**: Create the most basic line plot
- **Data**: Linear relationship (x: [1,2,3,4,5], y: [2,4,6,8,10])
- **Output**: Simple line connecting the points
- **Learning**: Introduction to `plt.plot()` and `plt.show()`

#### [plot2.py](./matplotlib/plot2.py) - Enhanced Line Plot
**Purpose**: Add labels, title, and styling to plots
- **Data**: Quadratic relationship (y = x²)
- **Features**: 
  - Red colored line
  - Axis labels ("X-axis", "Y-axis")
  - Plot title ("Basic Line Plot")
  - Legend showing equation
- **Learning**: Plot customization and annotation

#### [plot3.py](./matplotlib/plot3.py) - Multiple Line Plot
**Purpose**: Display multiple mathematical functions on same axes
- **Functions**:
  - Linear: y = x
  - Quadratic: y = x²
  - Cubic: y = x³
- **Features**: Automatic legend with function labels
- **Learning**: Comparing multiple datasets, legend usage

### Specialized Plot Types

#### [plot4.py](./matplotlib/plot4.py) - Scatter Plot
**Purpose**: Display data points as individual markers
- **Data**: Two arrays of related values
- **Styling**: Green circles ('o') with size 100
- **Features**: Custom marker size, color, and style
- **Applications**: Correlation analysis, data exploration

#### [plot5.py](./matplotlib/plot5.py) - Bar Chart
**Purpose**: Compare categorical data
- **Data**: Categories A-E with corresponding values [3,7,8,5,4]
- **Styling**: Orange bars
- **Applications**: Sales data, survey results, comparisons

#### [plot6.py](./matplotlib/plot6.py) - Histogram
**Purpose**: Show distribution of numerical data
- **Data**: 1000 random numbers from normal distribution
- **Features**: 30 bins, purple bars with black edges
- **Learning**: Statistical visualization, frequency distribution

#### [plot7.py](./matplotlib/plot7.py) - Pie Chart
**Purpose**: Show proportional data as circle segments
- **Data**: Four categories (Apples, Bananas, Cherries, Dates)
- **Features**: 
  - Percentage labels (`autopct="%1.1f%%"`)
  - Starting angle 90° for better presentation
- **Applications**: Market share, budget allocation

### Advanced Visualization

#### [plot8.py](./matplotlib/plot8.py) - Subplots
**Purpose**: Display multiple plots in single figure
- **Layout**: 1 row, 2 columns
- **Left Plot**: Red line plot of quadratic function
- **Right Plot**: Bar chart of same data
- **Features**: `plt.tight_layout()` for proper spacing
- **Learning**: Complex figure composition

#### [plot9.py](./matplotlib/plot9.py) - Customized Plot Styling
**Purpose**: Demonstrate advanced plot customization
- **Features**:
  - Blue color with dashed line style (`--`)
  - Circle markers (`o`) with size 8
  - Line width 2
  - Grid enabled for better readability
- **Learning**: Fine-tuned plot appearance

#### [plot10.py](./matplotlib/plot10.py) - 3D Surface Plot
**Purpose**: Visualize 3D mathematical functions
- **Function**: z = sin(√(x² + y²)) - creates ripple effect
- **Data**: 100x100 mesh grid from -5 to 5
- **Features**: 
  - 'viridis' colormap for surface coloring
  - 3D projection and surface rendering
- **Learning**: Three-dimensional data visualization

---

## 🔤 Natural Language Processing

This folder contains 13 examples covering fundamental NLP techniques from text preprocessing to advanced topic modeling.

### Text Preprocessing and Feature Extraction

#### [1.py](./NLP/1.py) - Text Preprocessing Pipeline
**Purpose**: Clean and normalize raw text data
- **Functions**:
  - `basic_clean()`: Removes URLs, emails, mentions, hashtags, special characters
  - `tokenize_stop_lemma()`: Tokenizes, removes stop words, applies lemmatization
- **Libraries**: spaCy for NLP, scikit-learn for stop words
- **Output**: List of clean, lemmatized tokens
- **Example**: "Emails like help@site.com..." → ['love', 'nlp', 'visit']

#### [8.py](./NLP/8.py) - Bag of Words (BoW)
**Purpose**: Convert text to numerical feature vectors
- **Input**: Small corpus of 4 sentences
- **Method**: `CountVectorizer` creates word frequency matrix
- **Output**: DataFrame showing word counts per document
- **Learning**: Foundation of text representation for ML

#### [9.py](./NLP/9.py) - TF-IDF Vectorization
**Purpose**: Weight words by importance using Term Frequency-Inverse Document Frequency
- **Input**: 3 documents about machine learning
- **Output**: TF-IDF matrix showing word importance scores
- **Advantage**: Reduces impact of common words, highlights distinctive terms

#### [10.py](./NLP/10.py) - Word2Vec Embeddings
**Purpose**: Create dense vector representations of words
- **Model**: Word2Vec with 50-dimensional vectors
- **Features**:
  - Skip-gram architecture (sg=1)
  - Window size 3, minimum count 1
- **Capabilities**: Find similar words, calculate word similarity
- **Example Output**: Most similar words to "learning"

### Classification and Machine Learning

#### [2.py](./NLP/2.py) - Text Classification with TF-IDF
**Purpose**: Sentiment classification using logistic regression
- **Dataset**: 10 movie reviews (positive/negative sentiment)
- **Pipeline**: TF-IDF vectorization + Logistic Regression
- **Features**: 
  - N-grams (1,2) for better context capture
  - Classification report with precision/recall
  - Probability scores for predictions
- **Applications**: Sentiment analysis, document categorization

#### [3.py](./NLP/3.py) - Hyperparameter Tuning
**Purpose**: Optimize model performance using grid search
- **Parameters Tuned**:
  - N-gram range: (1,1) vs (1,2)
  - Minimum document frequency: 1 vs 2
  - Analyzer: 'word' vs 'char_wb'
  - Regularization strength C: 0.25, 1.0, 4.0
- **Method**: 3-fold cross-validation with F1 scoring
- **Output**: Best parameters and cross-validation score

#### [11.py](./NLP/11.py) - Naive Bayes Classification
**Purpose**: Simple probabilistic text classifier
- **Algorithm**: MultinomialNB with Bag of Words
- **Dataset**: Tiny sentiment dataset (6 samples)
- **Features**: Fast training, works well with small datasets
- **Evaluation**: Classification report with metrics

### Advanced NLP Techniques

#### [4.py](./NLP/4.py) - Topic Modeling with LDA
**Purpose**: Discover hidden topics in document collection
- **Algorithm**: Latent Dirichlet Allocation (LDA)
- **Dataset**: 6 documents about pets and finance
- **Parameters**: 2 topics, 6 top words per topic
- **Output**: Topic words and document-topic distribution
- **Applications**: Content analysis, document organization

#### [13.py](./NLP/13.py) - Topic Modeling with Gensim
**Purpose**: Alternative topic modeling implementation
- **Library**: Gensim LDA model
- **Features**: More robust topic modeling with better parameter control
- **Output**: Formatted topic display with word weights

#### [5.py](./NLP/5.py) - Named Entity Recognition (NER) and POS Tagging
**Purpose**: Extract entities and analyze grammar structure
- **Library**: spaCy with English model
- **Capabilities**:
  - Named Entity Recognition (ORG, GPE, DATE, PERSON)
  - Part-of-Speech tagging
  - Lemmatization
  - Noun phrase chunking
- **Example**: Extracts "Apple" (ORG), "Bengaluru" (GPE), dates, etc.

### Text Analysis and Search

#### [6.py](./NLP/6.py) - Semantic Search
**Purpose**: Find relevant documents using cosine similarity
- **Method**: TF-IDF vectors + cosine similarity
- **Features**: 
  - N-gram range (1,2) for better matching
  - Ranked results by relevance score
- **Input**: Query string, returns top-k similar documents
- **Applications**: Document retrieval, recommendation systems

#### [7.py](./NLP/7.py) - Extractive Text Summarization
**Purpose**: Automatically summarize text by selecting important sentences
- **Algorithm**: 
  1. Calculate word frequencies (excluding stop words)
  2. Score sentences by sum of word frequencies
  3. Select top-scoring sentences in original order
- **Parameters**: Configurable number of output sentences
- **Applications**: News summarization, document analysis

#### [12.py](./NLP/12.py) - Document Similarity
**Purpose**: Measure similarity between text documents
- **Method**: TF-IDF vectorization + cosine similarity matrix
- **Output**: Symmetric similarity matrix showing all pairwise similarities
- **Applications**: Duplicate detection, content clustering

---

## 🔢 NumPy and Machine Learning

This folder contains 13 examples progressing from basic NumPy operations to machine learning algorithms.

### NumPy Fundamentals

#### [numpy1.py](./numpy_library/numpy1.py) - Array Creation
**Purpose**: Introduction to NumPy array creation methods
- **Array Types**:
  - 1D arrays from lists
  - 2D matrices
  - Zero matrices: `np.zeros((3,3))`
  - Ones matrices: `np.ones((2,4))`
  - Range arrays: `np.arange(0,10,2)`
  - Linear spacing: `np.linspace(0,1,5)`
- **Learning**: Foundation of NumPy data structures

#### [numpy2.py](./numpy_library/numpy2.py) - Array Operations
**Purpose**: Element-wise mathematical operations
- **Operations**: +, -, *, / (all element-wise)
- **Functions**: `np.sqrt()`, `np.power()`
- **Concept**: Vectorization - operations applied to entire arrays efficiently
- **Example**: [10,20,30,40] + [1,2,3,4] = [11,22,33,44]

#### [numpy3.py](./numpy_library/numpy3.py) - Array Indexing and Slicing
**Purpose**: Access and modify array elements
- **1D Operations**:
  - Single element access: `arr[0]`, `arr[-1]`
  - Slicing: `arr[0:3]`, `arr[::2]`
  - Element modification: `arr[2] = 99`
- **2D Operations**:
  - Element access: `mat[1,2]`
  - Row/column slicing: `mat[0,:]`, `mat[:,1]`

#### [numpy4.py](./numpy_library/numpy4.py) - Statistical Functions
**Purpose**: Calculate descriptive statistics
- **Functions**: `max()`, `min()`, `sum()`, `mean()`, `std()`
- **Index Functions**: `argmax()`, `argmin()` (return indices)
- **Input**: Array [3,7,2,9,5]
- **Applications**: Data analysis, feature engineering

#### [numpy5.py](./numpy_library/numpy5.py) - Random Number Generation
**Purpose**: Generate random data for testing and simulation
- **Functions**:
  - `randint(1,10,size=5)`: Random integers
  - `rand(5)`: Random floats [0,1]
  - `randn(3,3)`: Normal distribution matrix
  - `shuffle()`: In-place array shuffling
- **Applications**: Monte Carlo simulation, data augmentation

### Machine Learning Basics

#### [numpy6.py](./numpy_library/numpy6.py) - Train-Test Split
**Purpose**: Split dataset for model evaluation
- **Library**: scikit-learn's `train_test_split`
- **Dataset**: Simple X=[1,2,3,4,5,6,7,8], y=[2,4,6,8,10,12,14,16]
- **Split**: 80% training, 20% testing
- **Parameters**: `test_size=0.2`, `random_state=42` for reproducibility

#### [numpy7.py](./numpy_library/numpy7.py) - Linear Regression
**Purpose**: Predict continuous values using linear relationship
- **Algorithm**: Ordinary Least Squares regression
- **Data**: Perfect linear relationship y = 2x
- **Model Outputs**: 
  - Predictions for new values
  - Slope (coefficient): 2.0
  - Intercept: ≈0.0
- **Applications**: Price prediction, trend analysis

#### [numpy8.py](./numpy_library/numpy8.py) - Logistic Regression
**Purpose**: Binary classification (pass/fail prediction)
- **Data**: Study hours vs exam results
- **Algorithm**: Logistic regression with sigmoid function
- **Outputs**: 
  - Binary predictions (0 or 1)
  - Class probabilities for decision making
- **Applications**: Medical diagnosis, marketing response

#### [numpy9.py](./numpy_library/numpy9.py) - Decision Tree Classification
**Purpose**: Tree-based classification with interpretable rules
- **Dataset**: Iris flower dataset (150 samples, 4 features, 3 classes)
- **Parameters**: `max_depth=3` to prevent overfitting
- **Visualization**: Tree structure showing decision rules
- **Advantages**: Interpretable, handles non-linear relationships

#### [numpy10.py](./numpy_library/numpy10.py) - Feature Scaling
**Purpose**: Normalize features to same scale
- **Method**: StandardScaler (mean=0, std=1)
- **Example**: [[10,100], [20,200]] → normalized values
- **Importance**: Required for algorithms sensitive to feature scale (SVM, KNN, neural networks)

### Advanced Machine Learning

#### [numpy11.py](./numpy_library/numpy11.py) - K-Means Clustering
**Purpose**: Unsupervised grouping of similar data points
- **Algorithm**: K-Means with k=2 clusters
- **Dataset**: 6 points in 2D space forming 2 natural groups
- **Outputs**: 
  - Cluster centers (centroids)
  - Point labels indicating cluster membership
- **Visualization**: Scatter plot with colored clusters and red centroids

#### [numpy12.py](./numpy_library/numpy12.py) - Principal Component Analysis (PCA)
**Purpose**: Dimensionality reduction while preserving variance
- **Dataset**: Iris dataset (4D → 2D reduction)
- **Algorithm**: PCA extracts 2 most important components
- **Visualization**: 2D scatter plot colored by species
- **Applications**: Data visualization, noise reduction, compression

#### [numpy13.py](./numpy_library/numpy13.py) - Hierarchical Clustering
**Purpose**: Alternative clustering approach building tree of clusters
- **Algorithm**: Agglomerative clustering (bottom-up)
- **Dataset**: 6 points forming 2 natural groups
- **Method**: Starts with individual points, merges closest pairs
- **Advantages**: No need to specify number of clusters beforehand

---

## 🚀 Getting Started

### Prerequisites

Install the required packages:

```bash
pip install opencv-python matplotlib numpy scikit-learn spacy gensim
python -m spacy download en_core_web_sm
```

### Running the Examples

1. **Computer Vision**: Navigate to the folder and run any example:
   ```bash
   cd computer_vision
   python copy_1.py
   ```
   Note: All image examples now use sample images from the `resources/` folder (`man.jpg` and `mountainous.jpg`).

2. **Matplotlib**: Run plotting examples:
   ```bash
   cd matplotlib
   python plot1.py
   ```

3. **NLP**: Execute natural language processing examples:
   ```bash
   cd NLP
   python 1.py
   ```

4. **NumPy/ML**: Run numerical computing examples:
   ```bash
   cd numpy_library
   python numpy1.py
   ```

### Learning Path Recommendations

1. **Beginners**: Start with `numpy1.py` → `plot1.py` → `copy_3.py` → `1.py`
2. **Intermediate**: Focus on machine learning examples (`numpy7.py` - `numpy10.py`)
3. **Advanced**: Explore computer vision (`copy_11.py` - `copy_16.py`) and NLP (`2.py` - `7.py`)

### Common Issues and Solutions

- **Image path errors**: Update file paths in computer vision examples to match your system
- **Model download**: Ensure spaCy English model is installed for NLP examples
- **Display issues**: Some examples require GUI support for image/plot display
- **Memory errors**: Reduce data sizes in examples if running on limited memory systems

---

## 📖 Additional Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [spaCy Documentation](https://spacy.io/usage)
- [NumPy Reference](https://numpy.org/doc/stable/reference/)

---

*This repository serves as a comprehensive introduction to Python's most important data science and computer vision libraries. Each example is designed to be educational, practical, and easily extensible for your own projects.*