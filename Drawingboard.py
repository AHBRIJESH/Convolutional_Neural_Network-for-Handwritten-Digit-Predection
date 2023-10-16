import tkinter as tk
import numpy as np
from keras.models import load_model

# Load your pre-trained model (replace 'model.h5' with your model's path)
model = load_model('model.h5')

# Create a function to handle drawing on the canvas
def draw(event):
    x, y = event.x // 10, event.y // 10
    canvas.create_oval(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill="black")
    drawing_data[y, x] = 255  # Set pixel value to 255 when drawing

# Create a new tkinter window
root = tk.Tk()
root.title("Drawing Pad")

# Create a canvas widget
canvas = tk.Canvas(root, width=28 * 10, height=28 * 10, bg="white")
canvas.pack()

# Create a NumPy array to store the drawing data
drawing_data = np.zeros((28, 28), dtype=np.uint8)

# Bind the left mouse button to draw
canvas.bind("<B1-Motion>", draw)

# Create a function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    drawing_data.fill(0)
    prediction_label.config(text="Prediction: -")

# Create a clear button
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()

# Create a function to save the drawing as a 28x28 array and predict it
def predict_image():
    # Scale the pixel values from 0-255 and normalize to 0-1
    flattened_image = drawing_data.reshape(1, 784) / 255.0
    
    # Make a prediction using your model
    prediction = model.predict(flattened_image)
    
    # Display the prediction in a label
    predicted_class = np.argmax(prediction)
    prediction_label.config(text=f"Prediction: {predicted_class}")

# Create a predict button
predict_button = tk.Button(root, text="Predict Image", command=predict_image)
predict_button.pack()

# Create a label to show the prediction
prediction_label = tk.Label(root, text="Prediction: -")
prediction_label.pack()

# Run the tkinter main loop
root.mainloop()