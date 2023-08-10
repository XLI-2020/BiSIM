# Introduction
This is the implementation of this project. It contains two folders, namely differentiation and impuation, which corresponds to the two parts of paper
respectively.
# Requirements

- Pytorch 1.8.1
- Numpy 1.19.2
- Pandas 1.1.3
- Sklearn 0.24.1
- Matplotlib 3.3.2
- Shapely 1.8.1

You may use " pip3 install -r requirements.txt" to install the above libraries.


# Usage
**Step 1**: differentiate MARs and MNARs
``` 
cd ./differentiation ; python3 differentiator.py --site KDM --method DasaKM --thre 0.1
```
after this step, a csv file with differentiated results will be generated in the data folder.

**Step 2**: generate a json file for the input of BiSIM
``` 
cd ../imputation/preprocess ; python3 generate_input_json.py --site KDM --method DasaKM --thre 0.1 
```
An input file in json format will be generated in the data folder.

**Step 3**:  run BiSIM model for imputation
``` 
cd ../ ; nohup python3 -u main.py --site KDM --method DasaKM --thre 0.1 --epochs  500 --batch_size 32 > results.txt & 
```

# Explaination of Parameters
site: the building, e.g., KDM or WDS.

method:  the differentiator, e.g., DasaKM or TopoAC.

thre: the in-cluster differentiation threshold, e.g., thre=0, 0.1, 0.2.

batch_size: the number of samples for back propagation in one pass

epochs: the number of training rounds


# Acknowledgements

[//]: # (The whole dataset could be found from  [here]&#40;https://www.kaggle.com/c/indoor-location-navigation/data?select=train&#41;)

We appreciate the work of BRITS and SSGAN, and their contributed codes available in [here](https://github.com/caow13/BRITS) for BRITS and [here](https://github.com/zjuwuyy-DL/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation) for SSGAN.


# Citation

``` 
@article{chan2022continuous,
	title={Continuous Social Distance Monitoring in Indoor Space},
	author={Chan, Harry Kai-Ho and Li, Huan and Li, Xiao and Lu, Hua},
	journal={Proceedings of the VLDB Endowment},
	volume={15},
	number={7},
	year={2022},
	publisher={VLDB Endowment}
}
```




