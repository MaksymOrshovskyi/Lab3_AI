import tkinter as tk
from tkinter import filedialog, Label, Button, Scale, Text, Toplevel, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk


class FeatureExtractionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Feature Extraction Tool")

        # Зберігання зображень та векторів
        self.class_images = {'1': [], '2': [], '8': []}
        self.class_vectors = {'1': [], '2': [], '8': []}
        self.class_centroids = {'S1': {}, 'M1': {}}
        self.create_class_windows()

        # Елементи управління
        self.threshold_slider = Scale(master, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold")
        self.threshold_slider.set(128)
        self.threshold_slider.pack()

        self.segments_slider = Scale(master, from_=2, to=10, orient=tk.HORIZONTAL, label="Number of Segments")
        self.segments_slider.set(5)
        self.segments_slider.pack()

        Button(master, text="Upload Class 1 Images", command=lambda: self.upload_images('1')).pack()
        Button(master, text="Upload Class 2 Images", command=lambda: self.upload_images('2')).pack()
        Button(master, text="Upload Class 8 Images", command=lambda: self.upload_images('8')).pack()
        Button(master, text="Upload Unknown Image", command=self.upload_unknown_image).pack()
        Button(master, text="Classify Unknown Image", command=self.classify_image).pack()

        self.vector_text = Text(master, height=20)
        self.vector_text.pack(expand=True, fill='both')

        self.unknown_image_path = None
        self.unknown_vector = []

    def create_class_windows(self):
        self.class_windows = {}
        for class_name in ['1', '2', '8']:
            self.class_windows[class_name] = Toplevel(self.master)
            self.class_windows[class_name].title(f"Class {class_name} Images")
            self.class_windows[class_name].geometry("400x400")

    def upload_images(self, class_name):
        filepaths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if filepaths:
            self.class_images[class_name] = list(filepaths)
            self.process_class_images(class_name)

    def process_class_images(self, class_name):
        vectors = []
        for filepath in self.class_images[class_name]:
            vector, img_with_segments = self.process_image(filepath)
            if vector:
                vectors.append(vector)
                self.display_image(img_with_segments, self.class_windows[class_name])

        self.class_vectors[class_name] = vectors
        self.compute_centroids(class_name)

        for i, vector in enumerate(vectors):
            s1_vector = self.normalize_s1(vector)
            m1_vector = self.normalize_m1(vector)
            self.vector_text.insert(tk.END, f"Class {class_name} Оршовький {i + 1} S1: {s1_vector}\n")
            self.vector_text.insert(tk.END, f"Class {class_name} Оршовький {i + 1} M1: {m1_vector}\n")

    def process_image(self, filepath):
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None
        _, thresholded = cv2.threshold(img, self.threshold_slider.get(), 255, cv2.THRESH_BINARY)
        segments = self.segments_slider.get()
        height, width = thresholded.shape
        segment_width = width // segments
        vector = [np.sum(thresholded[:, i * segment_width:(i + 1) * segment_width] == 0) for i in range(segments)]

        img_color = cv2.cvtColor(thresholded, cv2.COLOR_GRAY2BGR)
        for i in range(1, segments):
            cv2.line(img_color, (i * segment_width, 0), (i * segment_width, height), (255, 0, 0), 1)
        img_pil = Image.fromarray(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        img_pil = img_pil.resize((100, 100))
        return vector, img_pil

    def display_image(self, img, window):
        tk_image = ImageTk.PhotoImage(img)
        label = Label(window, image=tk_image)
        label.image = tk_image
        label.pack(side=tk.LEFT)

    def normalize_s1(self, vector):
        total_sum = sum(vector)
        return [x / total_sum for x in vector] if total_sum > 0 else [0] * len(vector)

    def normalize_m1(self, vector):
        max_val = max(vector)
        return [x / max_val for x in vector] if max_val > 0 else [0] * len(vector)

    def compute_centroids(self, class_name):
        vectors = self.class_vectors[class_name]
        s1_vectors = [self.normalize_s1(v) for v in vectors]
        m1_vectors = [self.normalize_m1(v) for v in vectors]

        s1_centroid = np.mean(s1_vectors, axis=0)
        m1_centroid = np.mean(m1_vectors, axis=0)

        self.class_centroids['S1'][class_name] = s1_centroid
        self.class_centroids['M1'][class_name] = m1_centroid

        self.vector_text.insert(tk.END, f"\nClass {class_name} Оршовький S1Centr: {s1_centroid}\n")
        self.vector_text.insert(tk.END, f"Class {class_name} Оршовький M1Centr: {m1_centroid}\n \n")

    def upload_unknown_image(self):
        self.unknown_image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if self.unknown_image_path:
            self.unknown_vector, img_with_segments = self.process_image(self.unknown_image_path)
            if self.unknown_vector:
                self.display_image(img_with_segments, self.master)
                s1_vector = self.normalize_s1(self.unknown_vector)
                m1_vector = self.normalize_m1(self.unknown_vector)
                self.vector_text.insert(tk.END, f"\nUnknown Image Оршовький S1: {s1_vector}\n")
                self.vector_text.insert(tk.END, f"Unknown Image Оршовький M1: {m1_vector}\n")

    def load_unknown_image(self, image_path):
        # Load and preprocess the unknown image
        unknown_image = self.load_image(image_path)
        unknown_segments, unknown_s1, unknown_m1 = self.create_segments_and_vectors(unknown_image)

        # Store the feature vectors for classification
        self.unknown_image_data = {

            'Unknown Image Оршовький S1': unknown_s1,
            'Unknown Image Оршовький M1': unknown_m1
        }

    def classify_image(self):
        if not self.unknown_vector:
            messagebox.showerror("Error", "No unknown image loaded.")
            return

        s1_vector = self.normalize_s1(self.unknown_vector)
        m1_vector = self.normalize_m1(self.unknown_vector)

        distances = []
        self.vector_text.insert(tk.END, "\nDistances to centroids:\n")
        for class_name in ['1', '2', '8']:
            s1_dist = np.sum(np.abs(np.array(s1_vector) - self.class_centroids['S1'].get(class_name, [])))
            m1_dist = np.sum(np.abs(np.array(m1_vector) - self.class_centroids['M1'].get(class_name, [])))

            if s1_dist and m1_dist:
                distances.append((class_name, s1_dist, m1_dist))
                self.vector_text.insert(tk.END,
                                        f"Class {class_name}: S1 d = {s1_dist}, M1 d = {m1_dist}\n")

        if distances:
            distances.sort(key=lambda x: (x[1], x[2]))
            self.vector_text.insert(tk.END, f"\nClassification Result: {distances[0][0]} with minimum distance\n")
        else:
            self.vector_text.insert(tk.END, "Error: No valid distances computed.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = FeatureExtractionApp(root)
    root.mainloop()
