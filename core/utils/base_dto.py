# data transfer object
import dataclasses #효율적인 데이터 구조
from dataclasses import dataclass,_MISSING_TYPE
from munch import Munch #Munch dict업그레이드 버젼

expected = 'REQUIRED'
expected_train =  'REQUIRED_TRAIN'

def nested_dto(x,raw=False):
  return dataclasses.field(default_factory=lambda: x if raw else Munch.fromDict(x))
  # ** - raw이 True이면:** 입력값 그대로 기본값으로 사용
  # ** - raw이 False이면:** 입력값이 딕셔너리일 경우 Munch 객체로 변환하여 
@dataclass(frozen=True)
class Base: #모든 함수 Base값
  training : bool = None
  def __new__(cls,**kwargs):
    training  = kwargs.get('training',True) #트레이닝 없으면 true값 반환
    settablefields = cls.settablefields(**kwargs)  #classmethod
    mandatoryfields = cls.mandatoryfields(**kwargs)
    invalidkwargs = [ 
      {k: v} for k, v in kwargs.items() if k not in setteable_fields or v == EXPECTED or (v == EXPECTED_TRAIN and training is not False)
        ]
  # 모든 DTO가 Base DTO를 상속받으므로 코드 중복을 줄이고 재사용성을 높임
  # 변경가능한 필드값 아닌값
  @classmethod
  def settablefields(cls,**kwargs):
    pass
  @classmethod 
  def mandatoryfields(cls,**kwargs):
    pass
  
  @classmethod      #...=Base.fromdict(kwargs) kwargs값을 munch로 바꾼 instance 생성 
  def fromdict(cls,kwargs):
    for k in kwargs:
      if isinstance(kwargs[k],(list,tuple,dict)):
        kwargs[k] = Munch.fromDict(kwargs[k])
    return cls(**kwargs)
  #   fromdict 메서드는 딕셔너리를 입력받아 클래스 인스턴스를 생성합니다.
  # 딕셔너리 값 중 리스트, 튜플, 딕셔너리인 경우 Munch 객체로 변환합니다.
  # 변환된 딕셔너리는 클래스 생성자에게 전달되어 인스턴스가 생성됩니다.
  def to_dict(self):
          # selfdict = dataclasses.asdict(self) 
          selfdict = {}
          for k in dataclasses.fields(self):
              selfdict[k.name] = getattr(self, k.name)    #fromdict와 반대기능
              if isinstance(selfdict[k.name], Munch):
                  selfdict[k.name] = selfdict[k.name].toDict()
          return selfdict


  
