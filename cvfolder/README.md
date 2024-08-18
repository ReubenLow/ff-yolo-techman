# Installation
### 1. Clone the Repository
`git clone https://github.com/ReubenLow/ff-yolo-techman.git`

### 2. Set Up Virtual Environment
Create and activate a virtual environment in the root folder of cvfolder:\
`python -m venv .venv`\
`source .venv/bin/activate   # On Windows, use .venv\Scripts\activate`

### 3. Install Dependencies
Install the required Python packages:\
`pip install -r requirements.txt`

### 4. Set Up Environment Variable
`set PYTHONPATH=%PYTHONPATH%;<path-to-the-root-folder>`\

### 5. Locate the script
`cd examples/simple_scripts`\

### 6. Change the output directories of the coordinate and detected object JSON files in q_yolo_calibration.py.
![Screenshot from 2024-08-18 19-31-54](https://github.com/user-attachments/assets/01342568-561c-4e08-84ba-b4c22f692fe4)

### 7. Run the script
`python q_yolo_calibration.py`
