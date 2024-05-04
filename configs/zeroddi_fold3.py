
cudnn_benchmark = True
device = "cuda:0"
# model config
# zsl = "zsl"
model = dict(
    # model training and testing settings
    type='classifier',
    temperature=0.9,
    uni_lambda=0.7,
    UniformityLoss=True,
    leftmodel=dict(
        type='GNN_model',
        dropout=0.5,
        num_layer=2,
        JK='last',
        gnn_type='gin',
        Allfilename="./data/DrugBank5.1.9/DDI_final.csv",
        device=device,
        output_dim=256
    ),
    rightmodel=dict(
        type='BioTransEffectTextttention',
        output_dim=256
    ))

data = dict(
    train=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="train",
        file_dir="./data/DrugBank5.1.9/zsl",
        file_name="train.csv",
        output_file="./data/output/",
        zsl_mode="train",
        device=device,
        output_dim=256
    ),
    zsl_test=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="test",
        file_dir="./data/DrugBank5.1.9/zsl3",
        file_name="test.csv",
        output_file="./data/output/",
        zsl_mode="zsl",
        device=device,
        output_dim=256
    ),
    zsl_val=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="val",
        file_dir="./data/DrugBank5.1.9/zsl3",
        file_name="val.csv",
        output_file="./data/output/",
        device=device,
        zsl_mode="zsl",
        output_dim=256
    ),
    gzsl_test=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="test",
        file_dir="./data/DrugBank5.1.9/gzsl3",
        file_name="test.csv",
        output_file="./data/output/",
        device=device,
        zsl_mode="gzsl",
        output_dim=256
    ),
    gzsl_val=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="val",
        file_dir="./data/DrugBank5.1.9/gzsl3",
        file_name="val.csv",
        output_file="./data/output/",
        device=device,
        zsl_mode="gzsl",
        output_dim=256),
     val_seen=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="val",
        file_dir="./data/DrugBank5.1.9",
        file_name="val_seen.csv",
        output_file="./data/output/",
        device=device,
        zsl_mode="train",
        output_dim=256
    ),
    test_seen=dict(
        type='AttriTextBioBERTDataset',
        Allfilename='./data/DrugBank5.1.9/DDI_final.csv',
        mode="test",
        file_dir="./data/DrugBank5.1.9",
        file_name="test_seen.csv",
        output_file="./data/output/",
        device=device,
        zsl_mode="train",
        output_dim=256
    )

)

# yapf:enable
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluate_training_model = False
evaluate_during_training = False
max_grad_norm = 1.0,
parameter_averaging = True,
custom_hooks = [dict(type='NumClassCheckHook')]
weight_decay = 0.0
train_batch_size = 128
num_epochs = 100
gradient_accumulation_steps = 1
learning_rate = 0.0001
dist_params = dict(backsend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
save_max_model = True
use_single_cls = False
