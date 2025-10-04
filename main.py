import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import ImageTk, Image
import numpy as np
import matplotlib.pyplot as plt


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing Software")
        self.root.configure(bg="#222")  # Dark background

        # Frame for control buttons
        button_frame = tk.Frame(self.root, bg="#222")
        button_frame.pack(side="left", fill="y")

        # Buttons for each functionality
        button_opts = {'bg': '#333', 'fg': '#eee', 'activebackground': '#444', 'activeforeground': '#fff'}
        tk.Button(button_frame, text="Load Image", command=self.load_image, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Convert to Grayscale", command=self.convert_to_grayscale, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Spatial Filters", command=self.open_filter_window, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Contrast Adjustment", command=self.adjust_contrast, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Morphological Operations", command=self.morphological_operations, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Segmentation and Contours", command=self.segment_and_find_contours, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Custom Filters", command=self.open_custom_filter_window, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Display Histogram", command=self.display_histogram, **button_opts).pack(pady=5)
        tk.Button(button_frame, text="Equalize Histogram", command=self.equalize_histogram, **button_opts).pack(pady=5)

        # Canvas for image display
        self.original_canvas = tk.Canvas(self.root, width=400, height=400, bg="#111", highlightthickness=0)
        self.original_canvas.pack(side="left", padx=10, pady=10)
        self.processed_canvas = tk.Canvas(self.root, width=400, height=400, bg="#111", highlightthickness=0)
        self.processed_canvas.pack(side="left", padx=10, pady=10)

        # Label for image information
        self.info_label = tk.Label(self.root, text="", bg="#222", fg="#eee")
        self.info_label.pack(pady=5)

        self.original_image = None
        self.processed_image = None

    def load_image(self):
        """Load an image from the file system."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.original_image = cv2.imread(file_path)
            self.original_image = self.resize_image(self.original_image, 0.5)  # Resize the image to 50%
            self.display_image(self.original_image, self.original_canvas)
            self.display_image_info(self.original_image)

    def display_image(self, image, canvas):
        """Display the image on the Tkinter Canvas."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        image_tk = ImageTk.PhotoImage(image_pil)
        canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
        canvas.image = image_tk  # Store the reference to avoid garbage collection

    def display_image_info(self, image):
        """Display basic image information."""
        height, width, channels = image.shape
        info_text = f"Dimensions: {width}x{height}, Channels: {channels}"
        self.info_label.config(text=info_text)

    def convert_to_grayscale(self):
        """Convert the image to grayscale."""
        if self.original_image is not None:
            gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.processed_image = gray_image
            self.display_image(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

    def open_filter_window(self):
        """Open a new window to apply spatial filters."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Spatial Filters")
            window.configure(bg="#222")

            def apply_filter(filter_type):
                kernel_size = int(kernel_size_slider.get())
                if filter_type == "mean":
                    filtered_image = cv2.blur(self.original_image, (kernel_size, kernel_size))
                elif filter_type == "gaussian":
                    filtered_image = cv2.GaussianBlur(self.original_image, (kernel_size, kernel_size), 0)
                elif filter_type == "median":
                    filtered_image = cv2.medianBlur(self.original_image, kernel_size)
                elif filter_type == "laplacian":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)
                    filtered_image = cv2.convertScaleAbs(filtered_image)
                elif filter_type == "sobel":
                    gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                    sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=kernel_size)
                    sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=kernel_size)
                    filtered_image = cv2.convertScaleAbs(cv2.sqrt(sobelx ** 2 + sobely ** 2))
                self.processed_image = filtered_image
                self.display_image(filtered_image if len(filtered_image.shape) == 3 else cv2.cvtColor(filtered_image, cv2.COLOR_GRAY2BGR), self.processed_canvas)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="#222", fg="#eee", troughcolor="#333", highlightthickness=0)
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Mean Filter", command=lambda: apply_filter('mean'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Gaussian Filter", command=lambda: apply_filter('gaussian'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Median Filter", command=lambda: apply_filter('median'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Laplacian Filter", command=lambda: apply_filter('laplacian'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Sobel Filter", command=lambda: apply_filter('sobel'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)

    def adjust_contrast(self):
        """Adjust brightness and contrast using sliders."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Brightness/Contrast Adjustment")
            window.configure(bg="#222")

            def update(val):
                alpha = contrast_slider.get() / 100
                beta = brightness_slider.get() - 50
                adjusted_image = cv2.convertScaleAbs(self.original_image, alpha=alpha, beta=beta)
                self.display_image(adjusted_image, self.processed_canvas)

            contrast_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Contrast", command=update, bg="#222", fg="#eee", troughcolor="#333", highlightthickness=0)
            contrast_slider.set(50)
            contrast_slider.pack(pady=5)

            brightness_slider = tk.Scale(window, from_=0, to=100, orient=tk.HORIZONTAL, label="Brightness", command=update, bg="#222", fg="#eee", troughcolor="#333", highlightthickness=0)
            brightness_slider.set(50)
            brightness_slider.pack(pady=5)

    def resize_image(self, image, scale):
        """Resize the image to a smaller version."""
        width = int(image.shape[1] * scale)
        height = int(image.shape[0] * scale)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        return resized_image

    def apply_morphological_operation(self, operation, kernel_size):
        """Apply morphological operations like erosion, dilation, etc."""
        if self.original_image is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            if operation == 'erosion':
                processed_image = cv2.erode(self.original_image, kernel)
            elif operation == 'dilation':
                processed_image = cv2.dilate(self.original_image, kernel)
            elif operation == 'opening':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_OPEN, kernel)
            elif operation == 'closing':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_CLOSE, kernel)
            elif operation == 'gradient':
                processed_image = cv2.morphologyEx(self.original_image, cv2.MORPH_GRADIENT, kernel)
            self.display_image(processed_image, self.processed_canvas)

    def morphological_operations(self):
        """Open a new window to apply morphological operations."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Morphological Operations")
            window.configure(bg="#222")

            def apply(operation):
                kernel_size = int(kernel_size_slider.get())
                self.apply_morphological_operation(operation, kernel_size)

            kernel_size_slider = tk.Scale(window, from_=3, to=15, orient=tk.HORIZONTAL, label="Kernel Size", bg="#222", fg="#eee", troughcolor="#333", highlightthickness=0)
            kernel_size_slider.set(3)
            kernel_size_slider.pack(pady=5)

            tk.Button(window, text="Erosion", command=lambda: apply('erosion'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Dilation", command=lambda: apply('dilation'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Opening", command=lambda: apply('opening'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Closing", command=lambda: apply('closing'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)
            tk.Button(window, text="Gradient", command=lambda: apply('gradient'), bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)

    def segment_and_find_contours(self):
        """Segment the image and find contours."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Segmentation and Contours")
            window.configure(bg="#222")

            def apply_segmentation():
                threshold_value = threshold_slider.get()
                gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_image = self.original_image.copy()
                cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
                self.display_image(contour_image, self.processed_canvas)

            threshold_slider = tk.Scale(window, from_=0, to=255, orient=tk.HORIZONTAL, label="Threshold", bg="#222", fg="#eee", troughcolor="#333", highlightthickness=0)
            threshold_slider.set(128)
            threshold_slider.pack(pady=5)

            tk.Button(window, text="Apply", command=apply_segmentation, bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").pack(pady=5)

    def open_custom_filter_window(self):
        """Open a new window to apply a custom filter."""
        if self.original_image is not None:
            window = tk.Toplevel(self.root)
            window.title("Custom Filter")
            window.configure(bg="#222")

            def apply_filter():
                kernel = np.array([
                    [int(entry_00.get()), int(entry_01.get()), int(entry_02.get())],
                    [int(entry_10.get()), int(entry_11.get()), int(entry_12.get())],
                    [int(entry_20.get()), int(entry_21.get()), int(entry_22.get())]
                ], dtype=np.float32)
                custom_filtered_image = cv2.filter2D(self.original_image, -1, kernel)
                self.display_image(custom_filtered_image, self.processed_canvas)

            entries = []
            for i in range(3):
                row = []
                for j in range(3):
                    entry = tk.Entry(window, width=5, bg="#333", fg="#eee", insertbackground="#eee")
                    entry.grid(row=i, column=j, padx=5, pady=5)
                    row.append(entry)
                entries.append(row)

            entry_00, entry_01, entry_02 = entries[0]
            entry_10, entry_11, entry_12 = entries[1]
            entry_20, entry_21, entry_22 = entries[2]

            tk.Button(window, text="Apply Filter", command=apply_filter, bg="#333", fg="#eee", activebackground="#444", activeforeground="#fff").grid(row=3, columnspan=3, pady=10)

    def display_histogram(self):
        """Display the histogram of the image."""
        if self.processed_image is not None:
            # If the image is grayscale, show a single histogram
            if len(self.processed_image.shape) == 2:
                histogram = cv2.calcHist([self.processed_image], [0], None, [256], [0, 256])
                plt.plot(histogram, color='black')
                plt.xlim([0, 256])
            else:
                # Display histogram for each color channel
                for i, color in enumerate(['b', 'g', 'r']):
                    histogram = cv2.calcHist([self.processed_image], [i], None, [256], [0, 256])
                    plt.plot(histogram, color=color)
                    plt.xlim([0, 256])
            plt.show()

    def equalize_histogram(self):
        """Equalize the histogram of the image."""
        if self.processed_image is not None:
            if len(self.processed_image.shape) == 2:  # Grayscale
                equalized_image = cv2.equalizeHist(self.processed_image)
            else:  # Color image, convert to YCrCb and equalize the Y channel
                ycrcb = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                equalized_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            self.display_image(equalized_image, self.processed_canvas)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()
