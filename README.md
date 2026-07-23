# DM_underwater
This is the code of the paper "Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy" with some fixes.

usage steps:

- Install necessary Python packages from requirement.txt.
- Putting your data into the dataset folder. (There is some initial data).
- Download the pre-trained model, through this [link](https://drive.google.com/file/d/1As3Pd8W6XmQBU__83iYtBT5vssoZHSqn/view?usp=sharing). Then, put the model in the experiments_supervised folder.
- See 'config/underwater.json' to change parameters
- Execute infer.py to get the inference results in a new folder called experiments_val.
- Users can also comment and uncomment the line 13 and 14 in the config/underwater.json to change for the training process. And execute train.py for training.
- search_diffussion.py is used to search the sequence of time steps with the evolutionary algorithm. Users can use it in the inference process.


