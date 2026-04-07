A comprehensive research project evaluating the effectiveness of different deep learning architectures and audio feature representations (MFCCs vs. Mel Spectrograms) for automated speaker identification.

This project demonstrates that complex models (like our custom SE-ResCNN) achieve state-of-the-art accuracy when paired with high-dimensional Mel Spectrograms, while simpler models perform better with compressed MFCCs.

Dataset Information
Dataset Used: LibriSpeech ASR Corpus (train-clean-100 subset)
Description: A large-scale corpus of read English speech. For this project, a balanced subset of 30 to 75 distinct speakers was utilized to train and validate the models.
Download Link: [LibriSpeech Official Site (OpenSLR)](http://www.openslr.org/resources/12/train-clean-100.tar.gz)

Tech Stack and Libraries
This project is built using Python. The following libraries are required to run the pipeline:

TensorFlow / Keras: For building, compiling, and training the deep learning models (CNN, ResCNN, CRNN).

Librosa: For audio signal processing and extracting MFCCs and Mel Spectrograms.

Scikit-learn: For data splitting, label encoding, and calculating evaluation metrics (F1-score, Confusion Matrix).

NumPy & Pandas: For array manipulation and data structuring.

Matplotlib & Seaborn: For visualizing model performance and plotting confusion matrices.

Gradio: For deploying the real-time biometric "VoiceAuth" web interface.

tqdm & joblib: For progress tracking and parallel processing during heavy feature extraction.

Step-by-Step Implementation Guide
This project is optimized to run on Google Colab to leverage free cloud-based GPUs for faster model training.

Step 1: Google Colab Setup and Hardware Acceleration
Training deep neural networks on audio spectrograms is computationally expensive. You must enable a GPU accelerator before running the code.

Open Google Colab and upload the project notebook.

In the top menu bar, click on Runtime > Change runtime type.

Under the Hardware accelerator dropdown, select T4 GPU.

Click Save and then click Connect in the top right corner.

Step 2: Data Acquisition and Preprocessing
Raw audio cannot be fed directly into a CNN. It must be cleaned and standardized.

Action: The script connects to your Google Drive to locate the LibriSpeech dataset.

Preprocessing: The audio is loaded using librosa, converted to Mono, and resampled to 8000Hz. Silence at the beginning and end of the clips is trimmed.

Augmentation (Optional): Minor background noise is injected to ensure the model learns the voice, not the recording environment.

Step 3: Feature Extraction (The Core Logic)
The script converts the 1D audio waves into 2D visual representations.

MFCCs: Extracted to capture a compressed "vocal tract fingerprint" (shape: 20x80).

Mel Spectrograms: Extracted to capture rich, high-dimensional frequency details including pitch and harmonics (shape: 40x80).

Action: The script processes these features and caches them as .npy arrays to significantly speed up the training loop.

Step 4: Model Building and Compilation
The project constructs five different architectures to compare performance.

Action: The models (Standard CNN, Deep CNN, ResCNN, CRNN, and SE-ResCNN) are defined using Keras Functional API.

SE-ResCNN Details: The champion model uses Residual "shortcut" connections to prevent the vanishing gradient problem, combined with Squeeze-and-Excitation (SE) blocks to apply channel-wise attention to the most important audio frequencies.

Step 5: Model Training
The models are trained using the processed features.

Optimizer: Adam is used for adaptive learning rates.

Loss Function: Sparse Categorical Cross-Entropy is used as we are classifying mutually exclusive speaker IDs.

Optimization: EarlyStopping (patience=6) is implemented to automatically halt training when the validation loss stops improving, preventing the model from overfitting the data.

Step 6: Evaluation and Biometric UI
Once trained, the model evaluates unseen test data and generates performance metrics.

Action: The script outputs Accuracy, Macro F1-Scores, and plots a Confusion Matrix.

Gradio Deployment: Finally, a Gradio web interface is launched. This allows a user to select a "Claimed Identity", simulates a live audio feed, and outputs an "ACCESS GRANTED" or "DENIED" message based on the model's Softmax probability confidence.
