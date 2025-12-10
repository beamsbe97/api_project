To install dependencies, run ```pip install requirements.txt```
Modify conf/base.yml to train models with different configuration
In conf/base.yml, update all the folder paths under "train/build_dataset.folders" to where the training data is located

To start training, run ```python -m scripts.train --args.load conf/ablations/baseline.yml --save_path /path/to/training/result``` 

Weights(.pth files) and training logs are saved in "/path/to/training/result"

To run the Flask web application, switch to the "exp_1" branch
```git checkout exp_1```

Then run ```python app.py```