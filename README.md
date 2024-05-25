# kidneystonedetection
# Kidney Stone Detection Application

This application uses a machine learning model to detect kidney stones from medical images. It features a user-friendly graphical interface built with Tkinter and stores results in an SQLite database.

## Features

- Upload medical images for kidney stone detection.
- View detection results instantly.
- Store results in an SQLite database for future reference.

## Prerequisites

Before running the application, ensure you have the following installed on your system:

- *Windows OS* (The executable file is built for Windows)
- *Python 3.x* (if running from source)

## Download and Installation

### Download Executable

1. Go to the [Releases](https://github.com/yourusername/kidney-stone-detection/releases) page of this repository.
2. Download the latest kidney_stone_detection.exe file from the assets.

### Run the Executable

1. Navigate to the location where you downloaded the kidney_stone_detection.exe file.
2. Double-click the kidney_stone_detection.exe file to start the application.

### Download from Source

If you prefer to run the application from source, follow these steps:

1. Clone the repository:
    sh
    git clone https://github.com/yourusername/kidney-stone-detection.git
    cd kidney-stone-detection
    

2. Install the required Python packages:
    sh
    pip install -r requirements.txt
    

3. Run the application:
    sh
    python app.py
    

## Usage

1. *Launch the Application:*
   - If you downloaded the executable, double-click to open it.
   - If running from source, use the command python app.py.

2. *Upload an Image:*
   - Click the "Upload Image" button.
   - Select a medical image file from your computer.

3. *View Results:*
   - The application will process the image and display the detection result.
   - Results are automatically saved to the SQLite database.

## Project Structure

- app.py: Main application script with Tkinter GUI.
- kidney_stone_model.pkl: Pre-trained machine learning model.
- kidney_stone_results.db: SQLite database to store detection results.
- requirements.txt: List of Python dependencies.

## Contributing

Contributions are welcome! Please fork this repository and submit pull requests with your improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [scikit-image](https://scikit-image.org/)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Tkinter](https://wiki.python.org/moin/TkInter)

---

For any issues or questions, please open an issue on GitHub.