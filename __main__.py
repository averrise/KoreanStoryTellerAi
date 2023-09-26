import sys


if len(sys.argv) < 2: # 만약 sys.argv리스트가 길이가 2보다 작으면 즉 실행된 스크립트의 이름만 있는것이다
    print(
        f'[-] Only one commnad required which is in ("train", "inference", "interactive", "train_tokenizer)',
        file=sys.stderr,
    ) # 오류메시지 출력
    exit(-1) # 스크립트 종료

# 만약 sys.argv리스트가 길이가 1보다 크다면 스크립트는 할당 안받고 스크립트의 다음인덱스값은  command라는 변수에  arguments에는 나머지 값들 (스크립트: ex. python.py  )
_, command, *arguments = sys.argv

#ex) command가 train 인경우 train이라는 모듈에서 main,parser이라는 이름의 클래스나 함수를 가져온다. ("."은 해당 패키지를 의미한다.)
if command == "train":
    from .train import main, parser
elif command == "inference":
    from .inference import main, parser
elif command == "scoring":
    from .scoring import main, parser
#command에 해당하지 조건문에 해당하지 않는게 있다면 오류를 출력한다.
else:
    print(f'[-] Please type command in ("train", "inference")', file=sys.stderr)
    exit(-1)

exit(main(parser.parse_args(arguments)))  #main 함수의 반환값을 종료 상태 코드로 사용한다.

"""
## sys.argv
sys.argv는 sys모듈 내에 정의된 변수로, 명령 줄에서 실행된 스크립트에 전달된 인자들의 리스트를 담고 있습니다. (명령 프롬프트에서 직접 실행된 프로그램 혹은 코드 파일)

ex)

입력:$ python myscript.py arg1 arg2 arg3

sys.argv에 있는값은 ['myscript.py', 'arg1', 'arg2', 'arg3'] 이다.


##sys.stderr

sys.stderr는 표준 에러 스트림이다

sys.stderr 프로그램이 오류 메시지나 경고 메시지를 출력할때 사용됩니다.

오류나 경고 메시지를 일반 출력과 구분하여 처리할 수 있습니다.

##main(parser.parse_args(arguments))
parse_args는 ArgumentParser객체의 메서드로, 인자 리스트를 파싱하여 쉽게 접근 할 수 있는 객체로 변환한다



"""

