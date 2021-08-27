# KLUE_Topic_Classification_Competition
https://dacon.io/competitions/official/235747/overview/description

## Data Path 

```
​```
${KLUE_Folder(Dacon)}
├── train.py
├── inference.py
├── preprocess.py
├── utils.py
├── dataloader.py
|
├── saved_models
|   └── YOUR_MODLES.pth
|
├── submission
|   └── final_submission.csv
|
├── data
|   └── train_data.csv
|   └── test_data.csv
|   └── topic_dict.csv
|   └── sample_submission.csv
|
└── environment.yml
​```
```

## Environments Settings
- #### CUDA version >= 11.1
- #### Ubuntu 18.04
- #### huggingface
```
$ conda env create -n klue --file environment.yml
$ conda activate klue
```

## Preprocess Script

```bash
$ python preprocess.py --pt=klue/roberta-base --max_len=33
```

## Training Script
```bash
$ python train.py --gpu=0 --amp --pt=klue/roberta-base --epochs=10 --batch_size=32 --exp_name=yours
```

## Inference Script
```bash
$ python inference.py --gpu=0 --amp --pt=klue/roberta-base --batch_size=32 --model_path=YOUR_PATH
```

## TODO
- Create klue-ynat-only functions 
- Create environment.yml
- Create KLUE(ynat) Benchmark Results

## Result

