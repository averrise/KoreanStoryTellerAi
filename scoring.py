#커맨드 인자를 받기위한 argparse 코드 import
import argparse
#형태소 분석기를 가져오는 라이브러리
import Mecab
#nltk는 자연어 처리 작업에 유용한 다양한 도구와 라이브러리 제공
#sentence_blue는 두 개의 문장 또는 시퀀스 간 유사성을 평가하는 데 사용
#smoothingFunction은  blue스코어의 안정성을 향상시키기 위해서 사용된다
from nltk.translate.bleu_score import sentence_bleu, smoothingFunction은
#ROUGE스코어를 계산하는데 사용되는 패키지이다.
from rouge_metric import PyRouge

from src.data import jsonlload
from src.utils import get_logger

#파서는 커맨드로 "train을 받겠다"
parser = argparse.ArgumentParser(prog="train", description="Scoring Table to Text")
#추론 파일을 경로로 받겠다
parser.add_argument("--candidate-path", type=str, help="inference output file path")

#SCORE파일 메인 함수
def main(args):
    #SCORE 파악에 필요한 로거를 불러오기 위해 SCOREING을 인자로 전달하겠다.
    logger = get_logger("scoring")
    # Rouge 객체를 생성하고 Rouge-N 값을 설정합니다 (N-gram 단위로 유사성 평가)
    rouge = PyRouge(rouge_n=(1, 2, 4))
    # Mecab을 로드합니다. Mecab은 일본어 형태소 분석기입니다.
    logger.info(f'[+] Load Mecab from "/usr/local/lib/mecab/dic/mecab-ko-dic"')
    tagger = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ko-dic')
    # 데이터셋을 로드합니다.
    # 참조 데이터 (references)는 JSONL 파일에서 읽어온 "output" 필드 값들입니다.
    logger.info(f'[+] Load Dataset')
    references_j_list = jsonlload("resource/data/nikluge-2022-table-test-answer.jsonl")
    references = [j["output"] for j in references_j_list]
    # 후보 데이터 (candidate)는 사용자로부터 입력받은 후보 답변입니다.
    candidate_j_list = jsonlload(args.candidate_path)
    candidate = [j["output"] for j in candidate_j_list]
    # POS 태깅 시작을 로그에 기록합니다
    logger.info(f'[+] Start POS Tagging')

    # 참조 데이터에 대해 POS 태깅 수행
    for idx, sentences in enumerate(references):
        output = []
        for s in sentences:
            tokenized = []
            for mor in tagger.parse(s.strip()).split("\n"):
                if "\t" in mor:
                    splitted = mor.split("\t")
                    token = splitted[0]
                    tokenized.append(token)
            output.append(tokenized)
        references[idx] = output
    # 후보 데이터에 대해 태깅 수행
    for idx, s in enumerate(candidate):
        tokenized = []
        for mor in tagger.parse(s.strip()).split("\n"):
            if "\t" in mor:
                splitted = mor.split("\t")
                token = splitted[0]
                tokenized.append(token)
        candidate[idx] = tokenized
    

        # SmoothingFunction 객체를 생성하여 스무딩을 수행할 수 있는 함수를 설정합니다
    smoother = SmoothingFunction()

    # BLEU 스코어 초기화
    bleu_score = 0

    # 각 참조 문장에 대해 BLEU 스코어를 계산하고 누적하여 합산합니다
    for idx, ref in enumerate(references):
        # BLEU 스코어 계산 (1-gram에 대한 가중치만 사용)
        bleu_score += sentence_bleu(ref, candidate[idx], weights=(1.0, 0, 0, 0), smoothing_function=smoother.method1)

    # 계산된 BLEU 스코어를 참조 문장의 수로 나누어 평균을 계산합니다
    logger.info(f'BLEU Score\t{bleu_score / len(references)}')

    # ROUGE 스코어 계산
    # 후보 문장과 참조 문장 리스트를 ROUGE 평가 함수에 전달하여 ROUGE 스코어를 계산합니다
    rouge_score = rouge.evaluate(list(map(lambda cdt: " ".join(cdt), candidate)), \
                                list(map(lambda refs: [" ".join(ref) for ref in refs], references)))

    # ROUGE-1의 F1 스코어를 로그에 기록합니다
    logger.info(f'ROUGE Score\t{rouge_score["rouge-1"]["f"]}')

    # 프로그램이 메인으로 실행될 때만 실행되도록 합니다
    if __name__ == "__main__":
        exit(main(parser.parse_args()))

    