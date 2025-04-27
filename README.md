# Sentiment Analysis and Text Classification Pipeline

## Overview
This project implements a sentiment analysis pipeline using the BERT model from HuggingFace Transformers to classify movie reviews as positive or negative. Built on the IMDB dataset (public domain), the pipeline achieves an accuracy of approximately 85%. The system preprocesses text, fine-tunes BERT, stores results in MongoDB, and visualizes outcomes with Matplotlib. Developed as a personal project to explore Natural Language Processing (NLP), this work demonstrates skills in text classification and data pipelines, aligning with research interests in NLP and machine learning.

## Features
- **BERT Fine-Tuning**: Fine-tuned `bert-base-uncased` for binary sentiment classification on IMDB reviews.
- **Text Preprocessing**: Implemented tokenization and cleaning using NLTK and HuggingFace tokenizers.
- **Data Storage**: Stored classified reviews and sentiment scores in MongoDB for efficient retrieval.
- **Evaluation**: Achieved ~85% accuracy and F1-score on the test set, indicating robust performance.
- **Visualization**: Generated sentiment distribution and accuracy plots using Matplotlib.

## Technologies
- **Programming**: Python
- **Libraries/Frameworks**: HuggingFace Transformers, PyTorch, NLTK, Pandas, Scikit-learn, Matplotlib, PyMongo
- **Database**: MongoDB
- **Tools**: Git, VS Code, Anaconda, Jupyter Notebook
- **Dataset**: IMDB Movie Reviews (public domain)

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/SentimentAnalysis-NLP.git
   cd SentimentAnalysis-NLP
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   Dependencies include `transformers`, `torch`, `nltk`, `scikit-learn`, `pandas`, `pymongo`, and `matplotlib`.
3. **Download Dataset**:
   - Place the [IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25n/imdb-dataset-of-50k-movie-reviews) CSV in the `data/` directory.
4. **Set Up MongoDB**:
   - Install MongoDB locally or use a cloud instance.
   - Update `config.yaml` with your MongoDB connection string.

## Usage
- **Preprocessing**:
   - Run the preprocessing script:
     ```bash
     python preprocess.py
     ```
   - Outputs tokenized data in `data/processed/`.
- **Fine-Tuning**:
   - Train the BERT model:
     ```bash
     python train_bert.py
     ```
   - Or use `notebooks/train.ipynb` for interactive training.
   - Configure hyperparameters in `config.yaml` (e.g., epochs, batch size).
- **Evaluation**:
   - Evaluate model performance:
     ```bash
     python evaluate.py
     ```
   - Generates accuracy, F1-score, and visualizations in `notebooks/train.ipynb`.
- **Data Storage**:
   - Store results in MongoDB:
     ```bash
     python store_results.py
     ```

## Results
- **Accuracy**: ~85% on the test set, with an F1-score of ~84%.
- **Sample Output**:
  ```
  Review: "This movie was fantastic and thrilling!"
  Predicted Sentiment: Positive (Confidence: 0.92)
  ```
- **Visualization**: Sentiment distribution and accuracy plots are available in `notebooks/train.ipynb`, created with Matplotlib.

## Project Structure
```
SentimentAnalysis-NLP/
├── data/                 # Dataset (e.g., IMDB_Dataset.csv)
├── notebooks/            # Jupyter Notebooks
│   └── train.ipynb       # Training, evaluation, and visualization
├── preprocess.py         # Text preprocessing script
├── train_bert.py         # Training script
├── evaluate.py           # Evaluation script
├── store_results.py      # MongoDB storage script
├── config.yaml           # Hyperparameters and configuration
├── requirements.txt      # Dependencies
└── README.md             # Documentation
```

## Setup Requirements
- **Hardware**: Laptop with 8GB RAM (GPU optional; use Google Colab for faster training).
- **OS**: Windows/Linux/MacOS
- **Python**: Version 3.8+
- **Storage**: ~1GB for dataset and model weights
- **MongoDB**: Local or cloud instance

## Future Improvements
- Integrate a FastAPI endpoint for real-time sentiment prediction.
- Experiment with advanced models (e.g., RoBERTa) for improved accuracy.
- Add audio-based sentiment analysis using speech-to-text (e.g., Whisper).

## Acknowledgments
- Built as a personal project to advance skills in NLP and machine learning.
- Leveraged resources from HuggingFace tutorials, NLTK documentation, and open-source NLP guides.
