# Strikes detection

The goal of the project is to assist martial arts scoring judges by detecting
knockdowns - that is, when one of the fighters falls after an opponent's blow
which results in an immediate point deduction.

## Installation

1. Clone repo: `git clone https://github.com/dmarszalik/strikes-detection`
2. Create and activate virtual environment: 
    ```bash
    python -m venv venv
    source venv/bin/activate  # dla Linux/Mac
    venv\Scripts\activate  # dla Windows
    ```
3. Install dependencies: `pip install -r requirements.txt`

## Install dependencies from file setup.py

If you want to install the project in edit mode and be able to work on it, you can use the file `setup.py`. 

1. In the project directory, run: `pip install -e .`

## Project structure

1. Stage 1 - training the YOLOv8 model to recognize fighters, referees, gloves and shorts on fight footage.
The code and all files responsible for this part of the task can be found in the folder `./YOLOv8-custom/`. 
File `train.sh` The train.sh file is a script used to train YOLOv8 on labeled data. To run it, you need to:
### Linux/Mac

1. open a terminal in the project directory.

2. execute the following commands:

    ```bash
    chmod +x train.sh # Grant permissions to execute the script.
    ./train.sh # Run the script
    ```

### Windows (Bash on Windows)

1. Open a terminal in the project directory.

2. execute the following commands:

    ```bash
    bash train.sh # Run the script using Bash on Windows (may require installing tools such as Git Bash).
    ```
   
### Windows (PowerShell)

1. open a terminal in the project directory.

2. execute the following commands:

    ```powershell.
    .train.sh # Run the script in PowerShell.
    ```
   
---------------------------
2. Stage two, use the YOLO NAS POSE model to detect bboxes and the coordinates of points on the body for the detected
objects(poses), for each detected bbox run the previously trained YOLOv8-custom model and check whether the detected
person is a fighter or rather a referee. If a fighter then the program writes the poses to a numpy array and then saves
the prepared data to a csv file. This part of task can be found in the folder `./knock-down-detection/pose_detection.py`
3. A model was trained on such prepared data. After many experiments, the Random Forest Classifier model was selected.
This part of the project can be found in the file `./knock-down-detection/knock_down_detection.py`
4. The method of arriving at the final solution is shown in the notebooks in the folder `./experiments`
