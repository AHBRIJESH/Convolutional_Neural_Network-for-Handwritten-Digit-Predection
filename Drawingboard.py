import tkinter as tk
import numpy as np
from keras.models import load_model
model = load_model('model.h5')
def draw(event):
    x, y = event.x // 10, event.y // 10
    canvas.create_oval(x * 10, y * 10, (x + 1) * 10, (y + 1) * 10, fill="black")
    drawing_data[y, x] = 255  
root = tk.Tk()
root.title("Drawing Pad")
canvas = tk.Canvas(root, width=28 * 10, height=28 * 10, bg="white")
canvas.pack()
drawing_data = np.zeros((28, 28), dtype=np.uint8)
canvas.bind("<B1-Motion>", draw)
def clear_canvas():
    canvas.delete("all")
    drawing_data.fill(0)
    prediction_label.config(text="Prediction: -")
clear_button = tk.Button(root, text="Clear Canvas", command=clear_canvas)
clear_button.pack()
def predict_image():
    flattened_image = drawing_data.reshape(1, 784) / 255.0
    prediction = model.predict(flattened_image)
    predicted_class = np.argmax(prediction)
    prediction_label.config(text=f"Prediction: {predicted_class}")
predict_button = tk.Button(root, text="Predict Image", command=predict_image)
predict_button.pack()
prediction_label = tk.Label(root, text="Prediction: -")
prediction_label.pack()
root.mainloop()
