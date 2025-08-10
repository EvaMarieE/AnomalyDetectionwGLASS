# Path settings
DATASET_PATH="datasets/chest-x-ray"
AUG_PATH="datasets/chest-x-ray/train"
TRAIN_PATH="$DATASET_PATH/train/1"
TEST_PATH="$DATASET_PATH/test_anomaly"

cd ..
python main.py \
    --gpu 0 \
    --seed 1 \
    --test ckpt \
  net \
    -b resnetv2_101_21k \
    -le stages.0 \
    -le stages.1 \
    -le stages.2 \
    --pretrain_embed_dimension 1536 \
    --target_embed_dimension 1536 \
    --patchsize 2 \
    --meta_epochs 140 \
    --eval_epochs 1 \
    --dsc_layers 2 \
    --dsc_hidden 1024 \
    --pre_proj 1 \
    --mining 1 \
    --noise 0.015 \
    --radius 0.5 \
    --p 0.5 \
    --step 20 \
    --limit 392 \
  dataset \
     --imagesize 288 \
     chest \
    $DATASET_PATH $AUG_PATH \
    $DATASET_PATH/test_anomaly/anomaly


    #-b resnetv2_101_21k \
    #-le stages.0 \
    #-le stages.1 \
    #-le stages.2 \

    #-b wideresnet50 \
    #-le layer2 \
    #-le layer3 \
    