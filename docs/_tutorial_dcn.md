# Dense Correspondence Detection

## Generate Training Dataset for Dense Correspondence Network (DCN)

```shell
cd PATH_TO_ROBOT_VISION/robot_vision/dcn
python generate_dcn_train.py -d /media/gao/dataset/kvil/dustpan -n kvil -t /media/gao/dataset/train/dcn -o dustpan -s 0.6
```


## Training
```shell
cd PATH_TO_ROBOT_VISION/robot_vision/dcn
python train.py -p /media/gao/dataset/kvil/train/dustpan/20221226_155030
```

## Evaluation
to visualize the correspondences found by the trained model, run

```shell
cd PATH_TO_ROBOT_VISION/robot_vision/dcn
python viz/dcn_heatmap.py -p /media/gao/dataset/kvil/train/dustpan/20221226_155030
```