
import os

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from torch.optim.lr_scheduler import CyclicLR
from transformers import BartForConditionalGeneration



class StoryModule(pl.LightningModule):
    """
    Attributes:
        model: BART model
        total_steps: total training steps for lr scheduling
        max_learning_rate: Max LR
        min_learning_rate: Min LR
        warmup_rate: warmup step rate
        model_save_dir: path to save model
    """

    def __init__(
        self,
        model: BartForConditionalGeneration, #사용할 BART 모델
        total_steps: int, #전체 학습 스텝 수
        max_learning_rate: float,#학습률의 최대 값
        min_learning_rate: float,#학급률의 최소 값
        warmup_rate: float,#학습 초기에 학습률을 얼마나 빠르게 증가시킬 것
        model_save_dir: str,#모델을 저장할 디렉터리의 경로
    ):
        super().__init__()#부모 클래스인 LightningModule의 초기화 함수를 호출
        # 아래 변수들은 클래스 내부에서 사용될 변수들로, 초기화 함수의 인자로 받은 값을 그대로 저장합니다.
        self.model = model# BART 모델을 저장
        self.total_steps = total_steps# 전체 학습 스텝 수를 저장
        self.max_learning_rate = max_learning_rate# 학습률의 최대 값 저장
        self.min_learning_rate = min_learning_rate# 학습률의 최소 값 저장
        self.warmup_rate = warmup_rate # 학습률의 초기 증가 비율 저장
        self.model_save_dir = model_save_dir # 모델을 저장할 디렉터리 경로 저장

        # 모델의 설정 및 다른 하이퍼파라미터들을 저장
        # 이를 통해 나중에 모델을 로드할 때 이전에 사용한 설정을 그대로 불러올 수 있습니다.
        self.save_hyperparameters(
            {
                **model.config.to_dict(), # 모델의 설정을 딕셔너리 형태로 가져옴
                "total_steps": total_steps,#전체 학습 스텝 수
                "max_learning_rate": self.max_learning_rate, #학습률의 최대 값
                "min_learning_rate": self.min_learning_rate,#학습률의 최소 값
                "warmup_rate": self.warmup_rate,# 학습률의 초기 증가 비율
            }
        )
    # 학습 단계를 정의: 학습 데이터셋에 대해 실행되는 함수
    def training_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"], # 입력 문장의 토큰(단어) ID
            attention_mask=batch["attention_mask"], # 실제 문장 부분과 패딩 부분을 구분
            decoder_input_ids=batch["decoder_input_ids"], # 디코더에 주어지는 입력 (보통 타겟 문장의 시작 부분)
            decoder_attention_mask=batch["decoder_attention_mask"], # 디코더의 실제 문장 부분과 패딩 부분을 구분
            return_dict=True,
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1)#실제 정답 라벨을 준비
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])#모델의 예측값을 준비

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id)#손실(오차)를 계산함 : 모델의 예측값과 실제 정답 사이의 차이
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)#정확도를 계산함 : 얼마나 많은 예측이 실제 정답과 일치하는지

        metrics = {"loss": loss, "acc": accuracy} #손실과 정확도 값을 딕셔너리 형태로 저장
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True) #손실과 정확도 값 기록

        return metrics
    #검증단걔를 정의 : 검증 데이터셋에 대해 실행되는 함수
    def validation_step(self, batch, batch_idx):
        output = self.model(
            input_ids=batch["input_ids"], #입력 문장의 토큰(단어)ID
            attention_mask=batch["attention_mask"],#실제 문장 부분과 패딩 부분을 구분
            decoder_input_ids=batch["decoder_input_ids"],#디코더에 주어지는 입력 (보통 타겟 문장의 시작 부분)
            decoder_attention_mask=batch["decoder_attention_mask"],# 디코더의 실제 문장 부분과 패딩 부분을 구분
            return_dict=True, # 결과를 딕셔너리 형태로 반환
        )

        labels = batch["decoder_input_ids"][:, 1:].reshape(-1) #실제 정답 라벨 준비
        logits = output["logits"][:, :-1].reshape([labels.shape[0], -1])#모델의 예측값을 준비

        loss = F.cross_entropy(logits, labels, ignore_index=self.model.config.pad_token_id) # 손실(오차)를 계산합니다: 모델의 예측값과 실제 정답 사이의 차이
        accuracy = torchmetrics.functional.accuracy(logits, labels, ignore_index=self.model.config.pad_token_id)# 정확도를 계산합니다: 얼마나 많은 예측이 실제 정답과 일치하는지

        metrics = {"val_loss": loss, "val_acc": accuracy} # 손실과 정확도 값을 딕셔너리 형태로 저장합니다.
        self.log_dict(metrics, prog_bar=True, logger=True, on_epoch=True) # 손실과 정확도 값을 기록합니다.

        return metrics
    # 테스트 단계를 정의: 테스트 데이터셋에 대해 실행되는 함수
    def test_step(self, *args, **kwargs):
        return self.validation_step(*args, **kwargs)# 테스트 단계에서는 검증 단계와 동일한 로직을 사용합니다.
    #옵티마이저(학습 방법)와 학습률 스케줄러(학습률 조정 방법)을 설정하는 함수
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.max_learning_rate)#AdamW라는 학습 방법을 사용합니다.
        #CyclicLR :학습률을 주기적으로 조절하는 방법을 설정합니다.
        scheduler = CyclicLR(
            optimizer,
            base_lr=self.min_learning_rate,
            max_lr=self.max_learning_rate,
            step_size_up=int(self.total_steps * self.warmup_rate),# 언제까지 학습률을 증가시킬 것인지
            step_size_down=self.total_steps - int(self.total_steps * self.warmup_rate),# 그 이후에는 언제까지 학습률을 감소시킬 것인지

            mode='triangular', # 학습률 조정 방식을 'triangular' 방식으로 설정
            cycle_momentum=False# 모멘텀(가속도)는 사용하지 않습니다.
        )

        return {
            "optimizer": optimizer,# 사용할 옵티마이저
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"}, # 사용할 학습률 스케줄러
        }
    # 각 검증 epoch가 끝날 때 실행되는 함수
def validation_epoch_end(self, outputs):
    # 모든 GPU에서 생성된 출력들을 하나로 모읍니다.
    # 여러 GPU를 사용할 때, 각 GPU는 데이터의 일부분에 대해서만 계산을 수행하기 때문에 그 결과들을 모아야 합니다.
    outputs = self.all_gather(outputs)

    # 첫 번째 GPU에서만 아래의 코드를 실행합니다. 
    # 여러 GPU를 사용할 때, 중복된 작업을 피하기 위해 첫 번째 GPU만 결과를 처리하도록 합니다.
    if self.trainer.is_global_zero:
        # 각 GPU에서 생성된 손실값의 평균을 계산합니다.
        val_losses = [output["val_loss"].mean() for output in outputs]
        # 각 GPU에서 생성된 정확도의 평균을 계산합니다.
        val_accs = [output["val_acc"].mean() for output in outputs]

        # 모든 GPU의 결과를 기반으로 전체 데이터에 대한 손실값의 평균을 계산합니다.
        val_loss_mean = sum(val_losses) / len(val_losses)
        # 모든 GPU의 결과를 기반으로 전체 데이터에 대한 정확도의 평균을 계산합니다.
        val_acc_mean = sum(val_accs) / len(val_accs)

        # 현재의 epoch 번호, 전체 학습 스텝 수, 손실값, 정확도를 포함한 이름으로 모델을 저장합니다.
        # 이렇게 하면 나중에 모델의 성능 변화를 쉽게 파악할 수 있습니다.
        self.model.save_pretrained(
            os.path.join(
                self.model_save_dir,  # 모델을 저장할 디렉터리 경로
                f"model-{self.current_epoch:02d}epoch-{self.global_step}steps-{val_loss_mean:.4f}loss-{val_acc_mean:.4f}acc",
            ),
        )

