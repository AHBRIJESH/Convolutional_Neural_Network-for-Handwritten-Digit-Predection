# Neural Network Implementation with MNIST Dataset

(https://img.shields.io/badge/Implementation-Nural%20Network-darkred) 

This repository demonstrates the implementation of a Neural Network with a single layer using TensorFlow's Keras library. The model is trained on the MNIST dataset, consisting of 70,000 images. The dataset is split into two parts: the training set (`x_train`, `y_train`) and the test set (`x_test`, `y_test`).

## Files and Structure

- **Model.ipynb:**
  - Jupyter notebook containing the implementation of the Neural Network.
  - Uses TensorFlow's Keras library for building and training the model.
  - Converts 2D image arrays to 1D for compatibility with the Neural Network.
  - Includes a run method for easy execution.

- **Drawingboard.py:**
  - Python script to check predictions after running the model.
  - Allows users to interactively draw digits on a drawing board and receive predictions from the trained model.

## How to Use

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/neural-network-mnist.git
   cd neural-network-mnist
   ```

2. Open and run the `Model.ipynb` notebook to train the Neural Network and make predictions.

3. After running the notebook, execute the `Drawingboard.py` script to interactively draw digits and check the model's predictions.

   ```bash
   python Drawingboard.py
   ```

## Note

- The `Model.ipynb` notebook includes a run method that guides you through the training process. Follow the instructions in the notebook for seamless execution.

- The license and dependencies sections have been removed from this README for simplicity.

## Contribution

Contributions and suggestions are welcome! Feel free to open issues or submit pull requests to enhance the functionality or documentation of this project.

Happy coding! 😊🖌️🤖
