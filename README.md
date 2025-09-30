# Bird Sound Classification using Neural Networks
A deep learning project for classifying bird species based on their vocalizations using Convolutional Neural Networks (CNNs) and mel-spectrogram audio features.

This project implements an audio classification system that can identify bird species from their recorded calls. It processes audio recordings from the Xeno-Canto database, converts them into mel-spectrograms, and trains a CNN to classify them into 20 different bird species.Features

Audio Processing: Converts bird vocalizations into mel-spectrograms for neural network input
CNN Architecture: Custom convolutional neural network for audio classification
Top 20 Species: Focuses on the most well-represented bird species in the dataset
Data Augmentation: Handles variable-length audio through padding/truncation
Train/Val/Test Split: Proper 60/20/20 stratified data splitting
Visualization: Confusion matrix for performance analysis

Required libraries: 

pandas
numpy
librosa
tensorflow
scikit-learn
matplotlib
requests
tqdm

# Check directory before bash command
pip install pandas numpy librosa tensorflow scikit-learn matplotlib requests tqdm

# Project Structure
├── Main.py             
├── Main.ipynb           
├── download_audio.py      
├── BirdsVoice.csv       
├── audio/                 
└── README.md             

# License
This project uses data from Xeno-Canto, which is available under various Creative Commons licenses. Please respect the individual licenses of each recording.
Acknowledgments

# Acknowledgments 
Xeno-Canto for providing the bird sound database
The ornithology community for recording and sharing bird vocalizations
