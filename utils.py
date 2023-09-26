
import logging
import sys


def get_logger(name: str) -> logging.Logger: #-> logging.Logger은 함수가 반환할 값의 타입을 나타낸다. 즉 logging.Logger 타입의 값을 반환한다.
    """Return logger for logging 

    Args:
        name: logger name
    """
    logger = logging.getLogger(name) # 주어진 이름의 로거 객체를 가져옵니다
    logger.propagate = False # 로거의 propagate를 False로 설정하여, 로거가 상위로 이벤트를 전파하지 않도록 합니다
    logger.setLevel(logging.DEBUG) # 로거의 레벨을 DEBUG로 설정합니다
    """
    로깅 레벨
    CRITICAL: 50
    ERROR: 40
    WARNING: 30
    INFO: 20
    DEBUG: 10
    NOTSET: 0
    여기에서는 디버그 이므로 디버그부터 위에 있는 레벨의 메시지가 처리된다
    """
    #로거 핸들러가 설정되어 있지 않았다면
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout) # handler는 로그메시지를 스트림으로 출력하는데 사용한다
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))  #출력형식
        logger.addHandler(handler)  
    return logger

    """
    logger.addHandler(handler)

    logger: 로깅 이벤트를 생성하고 관리하는 객체이다.
    handler: 로그 메세지를 어디에 출력할지 정하는 변수이다
    addHandler(): logger가 handler에 의해 처리 될 수 있도록 하는 메서드이다.

    """
