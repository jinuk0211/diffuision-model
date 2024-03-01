import json
import subprocess
# #외부 프로그램 실행: 다른 프로그램을 Python 스크립트 내에서 실행하고 결과를 처리할 수 있습니다.
# 데이터 통신: 프로세스 간 데이터를 주고받을 수 있습니다.
# 병렬 처리: 여러 프로세스를 동시에 실행하여 작업 속도를 높일 수 있습니다.
# 자동화: 반복적인 작업을 자동화하기 위해 스크립트를 실행할 수 있습니다.
# 데이터 통신
# process = subprocess.Popen(["cat"], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
# process.stdin.write("Hello world!")
# output = process.stdout.read()
# print(output)
import yaml
import os
from .bucketeer import Bucketeer

# 입력 데이터 x의 json 키에 접근하여 데이터를 추출하고, rules에 정의된 규칙에 따라 필터링을 수행합니다.
# 규칙은 특정 조건을 확인하는 함수입니다.
# 필터링 과정:
# 입력 데이터 x에서 json 데이터를 추출합니다.
# rules 딕셔너리에 정의된 각 규칙을 반복적으로 적용합니다.
# 규칙의 키가 튜플일 경우, 규칙 함수에 여러 개의 json 값을 전달합니다.
# 규칙의 키가 튜플이 아닐 경우, 규칙 함수에 하나의 json 값을 전달합니다.
# 모든 규칙의 결과를 validations 리스트에 저장합니다.
# all(validations)를 통해 모든 규칙을 통과했는지 확인합니다.
# 모든 규칙을 통과했다면 True를 반환하고, 그렇지 않거나 오류 발생 시 False를 반환합니다.

class MultiFilter():
    def __init__(self, rules, default=False):
        self.rules = rules
        self.default = default

    def __call__(self, x):
        try:
            x_json = x['json']
            if isinstance(x_json, bytes):
                x_json = json.loads(x_json) 
            validations = []
            for k, r in self.rules.items():
                if isinstance(k, tuple):
                    v = r(*[x_json[kv] for kv in k])
                else:
                    v = r(x_json[k])
                validations.append(v)
            return all(validations)
        except Exception:
            return False

class MultiGetter():
    def __init__(self, rules):
        self.rules = rules

    def __call__(self, x_json):
        if isinstance(x_json, bytes):
            x_json = json.loads(x_json) 
        outputs = []
        for k, r in self.rules.items():
            if isinstance(k, tuple):
                v = r(*[x_json[kv] for kv in k])
            else:
                v = r(x_json[k])
            outputs.append(v)
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

def setup_webdataset_path(paths, cache_path=None):
    if cache_path is None or not os.path.exists(cache_path):
        tar_paths = []
        if isinstance(paths, str):
            paths = [paths]
        for path in paths:
            if path.strip().endswith(".tar"):
                # Avoid looking up s3 if we already have a tar file
                tar_paths.append(path)
                continue
            bucket = "/".join(path.split("/")[:3]) #아마존 웹 서비스(AWS)에서 제공하는 클라우드 스토리지 서비스입니다.
            # 웹 사이트 호스팅
            # 데이터 백업 및 복구
            # 콘텐츠 배포
            # 빅 데이터 분석
            # 모바일 애플리케이션 개발
            # 기타 다양한 클라우드 스토리지 요구 사
            result = subprocess.run([f"aws s3 ls {path} --recursive | awk '{{print $4}}'"], stdout=subprocess.PIPE, shell=True, check=True)
            files = result.stdout.decode('utf-8').split()
            files = [f"{bucket}/{f}" for f in files if f.endswith(".tar")]
            tar_paths += files

        with open(cache_path, 'w', encoding='utf-8') as outfile:
            yaml.dump(tar_paths, outfile, default_flow_style=False)
    else:
        with open(cache_path, 'r', encoding='utf-8') as file:
            tar_paths = yaml.safe_load(file)

    tar_paths_str = ",".join([f"{p}" for p in tar_paths])
    return f"pipe:aws s3 cp {{ {tar_paths_str} }} -"
