# Google Stock Price Prediction using RNN (LSTM)

This project demonstrates how to build and train a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers to predict Google stock prices.

## Project Steps

1.  **Setup and Data Download**:
    *   Install necessary libraries (`tensorflow`, `kaggle`).
    *   Configure Kaggle API credentials to download the dataset.
    *   Download the "Google Stock Prices Training and Test Data" dataset from Kaggle.

2.  **Data Loading and Preprocessing**:
    *   Load the training dataset (`Google_Stock_Price_Train.csv`) using pandas.
    *   Extract the 'Open' stock price column for training.
    *   Apply Feature Scaling (Min-Max Scaling) to normalize the training data between 0 and 1.

3.  **Data Structure Creation**:
    *   Create input sequences (`X_train`) and corresponding output values (`y_train`). Each input sequence consists of the previous 60 stock prices, and the output is the next stock price.
    *   Reshape the data to the required 3D format (samples, timesteps, features) for the LSTM model.

4.  **Building the RNN Model**:
    *   Initialize a Sequential Keras model.
    *   Add multiple LSTM layers with Dropout regularization to prevent overfitting.
    *   Add a Dense output layer to predict a single stock price value.
    *   Display the model summary.

5.  **Compiling and Training the RNN**:
    *   Compile the model using the Adam optimizer and Mean Squared Error loss function.
    *   Train the model on the prepared training data (`X_train`, `y_train`) for a specified number of epochs and batch size.

6.  **Preparing Test Data and Making Predictions**:
    *   Load the test dataset (`Google_Stock_Price_Test.csv`).
    *   Combine the training and test data to create input sequences for prediction on the test set, ensuring the model has access to the previous 60 days of stock prices leading up to the test period.
    *   Apply the same scaler used for training to the test input data.
    *   Create test sequences (`X_test`) in the required 3D format.
    *   Use the trained model to predict stock prices on the test data.
    *   Inverse transform the predicted prices to get them back to the original scale.

7.  **Visualization**:
    *   Plot the real Google stock prices from the test set.
    *   Plot the predicted Google stock prices.
    *   Add a title, axis labels, and a legend to the plot for clarity.
    *   Display the visualization to compare the real and predicted prices.

## Requirements

*   Python 3.x
*   `tensorflow`
*   `kaggle`
*   `pandas`
*   `numpy`
*   `matplotlib`
*   `scikit-learn`

## Dataset

The dataset used in this project is "Google Stock Prices Training and Test Data" from Kaggle. It contains historical Google stock prices.

## How to Run

1.  Ensure you have Python and the required libraries installed.
2.  Download the notebook and the dataset files.
3.  Obtain a Kaggle API key and place it in a `kaggle.json` file in the appropriate directory as shown in the notebook.
4.  Run the cells in the notebook sequentially.
