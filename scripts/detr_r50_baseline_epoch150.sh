script_name1=`basename $0`
script_name=${script_name1:0:${#script_name1}-3}

pip install termcolor --user
MKL_THREADING_LAYER=GNU python -m torch.distributed.launch \
    --nproc_per_node=8 \
    main.py \
    --coco_path ../data/coco \
    --output_dir output/$script_name