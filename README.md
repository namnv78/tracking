# tracking
0. Datasets: 
- https://drive.google.com/drive/folders/1kbk8KFXlPjZ8YqF2J9Ef-F2ZSLbUJ0og?usp=sharing
- [Data](https://drive.google.com/file/d/1ctUGrzJuRcax6ZgdtzE_gsBarPdN-uIz/view?usp=sharing) for training trajectory embedding model

1. Download and extract a [modified snapshot](https://drive.google.com/file/d/1c3gn_0n_UVLRAmyMU9jdn3wgJgmS61Um/view?usp=sharing) of [yolov5](https://github.com/ultralytics/yolov5) into `src/yolov5/`
2. Pretrained weights can be downloaded:
- Yolov5 detector: 
    - [Vehicle](https://drive.google.com/file/d/1EZ7ls95GGUi5QjCGTtdmlKUi65Ox6HyS/view?usp=sharing) 
    - [Pedestrian](https://drive.google.com/file/d/1w65gH2n0Tkn8Y9GCFBVhuTMyUUuEL9cp/view?usp=sharing)
- Appearance re-id: 
    - [Vehicle](https://drive.google.com/file/d/1sjVBtDZsVdSe5BTxj8EEQn7qVSQN8UG0/view?usp=sharing) 
    - [Pedestrian](https://drive.google.com/file/d/1jSYQ-as1mgSL7lV0GU1bQKQi4wm9bq_u/view?usp=sharing)
- Trajectory model: [here](https://drive.google.com/file/d/1LyK8FevMrt2lBMugpMEgmL8HcmmY0giB/view?usp=sharing)
3. Notebooks
- For training the trajectory embedding model, [refer this notebook](/https://github.com/namnv78/tracking/blob/main/src/notebooks/TrajectoryTrain.ipynb)
- For inferencing normal videos, [refer this notebook](https://github.com/namnv78/tracking/blob/main/src/notebooks/InferenceOnline.ipynb)
- For inferencing MOT videos, [refer this notebook](https://github.com/namnv78/tracking/blob/main/src/notebooks/InferenceOnlineMOT.ipynb)