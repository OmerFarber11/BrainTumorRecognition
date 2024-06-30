import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import numpy as np
import os
import matplotlib.pyplot as plt

IMAGE_SHAPE = (128, 128)

from AI import DLNetwork, DLNeuronsLayer

class BrainCancerPredictor(tk.Tk):
    def __init__(self):
        """
        Initializes the BrainCancerPredictor application.
        Sets up the main window, neural network, and GUI widgets.
        """
        super().__init__() # for tkinter
        self.title("Brain Cancer Predictor")
        self.geometry("600x700")
        self.configure(bg='#f0f0f0')  # Set background color

        #creates the network with all the layers
        self.network = self.create_network()
        
        #gets the parameters - the bias and weight for every neuron
        self.network.load_parameters("C:/Users/omifa/Downloads/BrainProject")  # Load saved model parameters

        self.create_widgets()
        self.style_widgets()

    #this function creates and returns a neural network network base
    def create_network(self):
        classifications = ['Giloma', 'Meningioma', 'No Tumor', 'Pituitary']

        network_layers = [
            DLNeuronsLayer(np.prod(IMAGE_SHAPE), 32, "leaky_relu", 0.01, 1),
            DLNeuronsLayer(32, 16, "relu", 0.001, 2),
            DLNeuronsLayer(16, len(classifications), "softmax", 0.0005, 3)
        ]
        
        return DLNetwork(network_layers)

    #creates and packs the widgets (buttons, labels, canvas) for the GUI.
    def create_widgets(self):
        # main frame
        self.main_frame = ttk.Frame(self, padding=10)
        self.main_frame.pack(pady=20, padx=20, fill='both', expand=True)

        # upload button
        self.upload_button = ttk.Button(self.main_frame, text="Upload Patient's Brain", command=self.upload_image)
        self.upload_button.grid(row=0, column=0, pady=10, padx=10)

        # image display
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(row=1, column=0, pady=10, padx=10)

        # result display
        self.result_label = ttk.Label(self.main_frame, text="", font=('Helvetica', 16, 'bold'))
        self.result_label.grid(row=2, column=0, pady=10, padx=10)

        # canvas
        self.plot_frame = ttk.Frame(self.main_frame)
        self.plot_frame.grid(row=3, column=0, pady=10, padx=10)
        self.plot_canvas = tk.Canvas(self.plot_frame, width=400, height=300, bg='white', bd=2, relief='sunken')
        self.plot_canvas.pack()

    def style_widgets(self):
        """
        Styles the widgets using ttk.Style.
        """
        style = ttk.Style()
        style.configure('TButton', font=('Helvetica', 14), padding=10)
        style.configure('TLabel', font=('Helvetica', 12))
        
        # Custom frame and label styles
        style.configure('TFrame', background='#f0f0f0')
        style.configure('TLabel', background='#f0f0f0', font=('Helvetica', 12))

    #opens a file dialog for the user to upload an image, and processes the image for display and prediction.
    def upload_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = Image.open(file_path).convert("L")
            image = image.resize(IMAGE_SHAPE)
            self.display_image(image)
            self.predict(file_path)

    #Displays the uploaded image in the GUI.
    def display_image(self, image):
        img = ImageTk.PhotoImage(image)
        self.image_label.configure(image=img)
        self.image_label.image = img

    #This function gets an image (PIL.Image) and preprocesses the image for prediction by the neural network.
    def preprocess_image(self, file_path):
        image = Image.open(file_path).convert('L')  # Convert to grayscale
        image = image.resize((128, 128))
        image = np.array(image)
        image = image / 255.0  # Normalize pixel values
        image = image.reshape((128* 128 ,1))
        return image

    #This function gets the brain image (PIL.Image) and predicts what type of tumor is in the brain.
    def predict(self, image):
        processed_image = self.preprocess_image(image)
        
        print(processed_image.shape)
        #using the AI to predict
        prediction = self.network.forward_propagation(processed_image)
        
        class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
        self.update_result_label(prediction, class_names)
        self.plot_prediction_distribution(prediction, class_names)

    #This function gets the prediction probabilities and the tumors types and updates the "winner" - highest probability result.
    def update_result_label(self, prediction, class_names):
        class_idx = np.argmax(prediction)
        result_text = f"Prediction: {class_names[class_idx]}"
        self.result_label.config(text=result_text)

    #This function gets the prediction distribution and the tumors type and creates a column chart showing the probabilites.
    def plot_prediction_distribution(self, prediction, class_names):
        
        # clears the canvas
        self.plot_canvas.delete("all")
        
        # gets probabilities from prediction
        probabilities = prediction.flatten()
        
        # creates a bar chart
        self.plot_canvas.create_text(200, 20, text="Prediction Distribution", font=('Helvetica', 14))
        bar_width = 40
        max_prob = np.max(probabilities)
        for i, (prob, class_name) in enumerate(zip(probabilities, class_names)):
            bar_height = prob * 200 / max_prob  # Scale height based on maximum probability for visualization
            self.plot_canvas.create_rectangle(50 + i * 100, 250, 50 + i * 100 + bar_width, 250 - bar_height, fill='blue')
            self.plot_canvas.create_text(50 + i * 100 + bar_width / 2, 270, text=class_name, anchor='n')

#main
if __name__ == "__main__":
    app = BrainCancerPredictor()
    app.mainloop()
