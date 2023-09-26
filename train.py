import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import BartForConditionalGeneration, AutoTokenizer

from src.data import StoryDataLoader
from src.module import StoryModule
from src.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts") #결과물을 저장할 디렉토리의 경로를 정할 수 있다
g.add_argument("--model-path", type=str, default="gogamza/kobart-base-v2", help="model file path")
g.add_argument("--tokenizer", type=str, default="gogamza/kobart-base-v2", help="huggingface tokenizer path")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
parser.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--max-learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--min-learning-rate", type=float, default=1e-5, help="min Learning rate")
g.add_argument("--warmup-rate", type=float, default=0.1, help="warmup step rate")
g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--evaluate-interval", type=int, default=500, help="validation interval")
g.add_argument("--seed", type=int, default=42, help="random seed")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")
# fmt: on


def main(args):
    logger = get_logger("train")                                    # 로깅을 위한 logger 객체를 얻는다

    os.makedirs(args.output_dir)                                    #지정된 경로에 디렉토리를 생성합니다.
    
    logger.info(f'[+] Save output to "{args.output_dir}"')          # 생성한 디렉토리 경로를 로깅합니다.
    
    #사용자로부터 입력받은  모든 인자들에 대한 정보를 로그로 남긴다
    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")
    
    # 랜덤 시드를 설정한다
    logger.info(f"[+] Set Random Seed to {args.seed}")              # 랜덤시드값 정보 기록
    pl.seed_everything(args.seed)                                   #훈련의 재현성을 보장 -> 즉 무작위 분할과 모델가중치 초기화때문에 매번 실행할때마다 결과가 다르게 나오는 것을 막는다.
    
    #GPU
    logger.info(f"[+] GPU: {args.gpus}")                            #GPU값 정보기록

    #토크나이저 불러오기
    logger.info(f'[+] Load Tokenizer"')                          #load Tokenizer한거 정보기록
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)    

    #데이터셋 load하기
    logger.info(f'[+] Load Dataset')
    train_dataloader = StoryDataLoader("resource/data/nikluge-sc-2023-train.jsonl", tokenizer, args.batch_size, args.max_seq_len) 
    valid_dataloader = StoryDataLoader("resource/data/nikluge-sc-2023-dev.jsonl", tokenizer, args.valid_batch_size, args.max_seq_len)
    
    
    #전체 훈련 스텝을 계산한다
    total_steps = len(train_dataloader) * args.epochs             

    # 모델 경로가 지정되어 있으면 해당 경로에서 모델을 불러온다
    if args.model_path:
        logger.info(f'[+] Load Model from "{args.model_path}"')
        model = BartForConditionalGeneration.from_pretrained(args.model_path)

    # pytorch Lightning 모둘을 불러온다.
    logger.info(f"[+] Load Pytorch Lightning Module")
    lightning_module = StoryModule(
        model,
        total_steps,
        args.max_learning_rate,
        args.min_learning_rate,
        args.warmup_rate,
        args.output_dir
    )
    
    # 훈련을 시작한다.
    logger.info(f"[+] Start Training")
    train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")] # 첫번째 인자 : TensorBoard(결과물)이 저장될 디렉퇴 경로 
    if args.wandb_project:          # args.wandb_project은 아마 가중치와 바이어스값을 가진 변수 일 것이다
        train_loggers.append(
            WandbLogger(
                name=args.wandb_run_name or os.path.basename(args.output_dir),
                project=args.wandb_project,
                entity=args.wandb_entity,
                save_dir=args.output_dir,
            )
        )

    # If evaluate_interval passed float F, check validation set 1/F times during a training epoch
    if args.evaluate_interval == 1:
        args.evaluate_interval = 1.0
    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.evaluate_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[LearningRateMonitor(logging_interval="step")],
        gpus=args.gpus,
    )
    trainer.fit(lightning_module, train_dataloader, valid_dataloader)


if __name__ == "__main__":
    exit(main(parser.parse_args()))


'''
argparse: 파이썬 표준 라이브러리의 일부로, 명령 줄 인자 파싱을 도와주는 라이브러리입니다.

os: 파이썬 표준 라이브러리로, 운영 체제와 상호 작용하기 위한 기능을 제공합니다.

pytorch_lightning: PyTorch를 위한 확장 프레임워크로, 딥러닝 모델 학습과 관련된 보일러플레이트 코드를 줄여줍니다.

transformers: HuggingFace에서 제공하는 라이브러리로, 다양한 사전 훈련된 NLP 모델에 대한 인터페이스를 제공합니다.

src.data, src.module, src.utils: 사용자가 정의한 모듈로, 특정 데이터 로더, 모델 모듈, 유틸리티 함수가 포함될 것으로 추정됩니다.


Wandb Options: wandb는 Weights & Biases라는 도구의 약자로,

딥러닝 모델의 학습 과정을 모니터링하거나 로깅하는 데 사용되는 서비스입니다.

이 서비스와 관련된 옵션을 설정하는 인자들이 추가되어 있습니다.


#logging

프로그램이 실행되는 동안 발생하는 이벤트나 메시지, 에러 정보 등을 시간 순서대로 기록하는 행위나 그 기록 자체를 의미한다


'''
