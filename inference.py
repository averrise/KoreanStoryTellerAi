
#argparse를 이용하여 인자를 파싱할 준비를 함
import argparse
#tqdm은 반복문의 상황을 시각적으로 표시
from tqdm import tqdm

#파이토치 임포트
import torch
#트랜스포머 라이브러리에서  BartForConditionalGeneration 및 AutoTokenizer 클래스를 임포트
from transformers import BartForConditionalGeneration, AutoTokenizer

#상대 경로에 있는 src/data 모듈에서 StoryDataLoader, jsonlload, jsonldump 함수를 임포트합니다.
from src.data import StoryDataLoader, jsonlload, jsonldump
#밑에도 마찬가지
from src.utils import get_logger


#argumentParser를 통해서 명령줄에서 train을 받으면 Paser를 실행함(여러 명령어를 넣은 기계? 라고 보면됌)
parser = argparse.ArgumentParser(prog="train", description="Inference Table to Text with BART")

#parser는 인자값으로 모델 경로, 토크나이저, 성과 경로(뭔지 모름), 훈련 배치 사이즈, 최대 모델이 읽을 수 있는 길이
#최대 요약 길이, num_beams는 각 생성 단계에서 유지할 후보 문장의 수, 기본 수행 장치를 스크립트 명령어로 받는다.
parser.add_argument("--model-ckpt-path", type=str, help="Table to Text BART model path")
parser.add_argument("--tokenizer", type=str, required=True, help="huggingface tokenizer path")
parser.add_argument("--output-path", type=str, required=True, help="output tsv file path")
parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
parser.add_argument("--max-seq-len", type=int, default=512, help="max sequence length")
parser.add_argument("--summary-max-seq-len", type=int, default=64, help="summary max sequence length")
parser.add_argument("--num-beams", type=int, default=3, help="beam size")
parser.add_argument("--device", type=str, default="cpu", help="inference device")


#메인 함수 설명
def main(args):
    #추론에 loggar 소환 즉 추론에서 어떤일이 벌어지는지 확인하겠다는 뜻
    logger = get_logger("inference")
    #로그를 사용하여 디바이스 정보를 출력
    logger.info(f"[+] Use Device: {args.device}")
    #지정된 장치로 설정하기 위해 torch.device를 사용하여 디바이스를 선택
    device = torch.device(args.device)

    #로그로 토그나이저 정보 출력 AND 사전 훈련 토크나이저 로드
    logger.info(f'[+] Load Tokenizer from "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    #로거로 데이서 셋 정보 출력
    logger.info(f'[+] Load Dataset')
    dataloader = StoryDataLoader("resource/data/nikluge-sc-2023-test-answer.jsonl", tokenizer=tokenizer, batch_size=args.batch_size, max_length=args.max_seq_len, mode="infer")
    #loggar로 모델 어디서 가져왔는지 출력
    logger.info(f'[+] Load Model from "{args.model_ckpt_path}"')
    #사전 훈련된 BART 모델의 가중치를 해당 경로에서 로드하여 BartForConditionalGeneration 모델 객체를 생성
    model = BartForConditionalGeneration.from_pretrained(args.model_ckpt_path)
    model.to(device)

    #로그를 사용하여 평가 모드로 설정하고 그라디언트를 비활성화하는 것을 출력합니다.
    logger.info("[+] Eval mode & Disable gradient")
    #모델을 평가모드로 전환
    model.eval()
    #그레디언트를 비활성화
    torch.set_grad_enabled(False)

    #로그를 사용하여 추론 시작을 출력합니다
    logger.info("[+] Start Inference")
    total_summary_tokens = []
    #DATALOADER에서 batch만큼가져옴
    for batch in tqdm(dataloader):
        #텍스트 토큰의 id를 device로 전달
        dialoge_input = batch["input_ids"].to(device)
        #특정 토큰에 대한 유의를 표시
        attention_mask = batch["attention_mask"].to(device)
        print(dialoge_input)


        #model.generate를 사용하여 요약을 생성
        summary_tokens = model.generate(
            dialoge_input,
            attention_mask=attention_mask,
            #디코더의 시작 토큰 아이디
            decoder_start_token_id=tokenizer.bos_token_id,
            #요약의 최대길이
            max_length=args.summary_max_seq_len,
            #패팅 토큰의 아이디
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            #종료 토큰의 아이디
            eos_token_id=tokenizer.eos_token_id,
            num_beams=args.num_beams,
            use_cache=True,
        )
        total_summary_tokens.extend(summary_tokens.cpu().detach().tolist())
    
    #로그로 디코딩 정보 표현
    logger.info("[+] Start Decoding")
    #tokenizer.decode를 사용하여 토큰을 원래의 문장으로 디코딩하고, skip_special_tokens=True를 통해 특수 토큰들을 무시
    decoded = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in tqdm(total_summary_tokens)]
    j_list = jsonlload("resource/data/nikluge-sc-2023-test-answer.jsonl")
    
    for idx, oup in enumerate(decoded):
        j_list[idx]["output"] = oup

    jsonldump(j_list, args.output_path)


if __name__ == "__main__":
    exit(main(parser.parse_args()))
