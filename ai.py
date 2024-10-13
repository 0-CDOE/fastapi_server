import logging
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
import logging  # 로그 출력을 위한 모듈
from pathlib import Path
# 각 단계별 클래스를 개별적으로 임포트합니다.
from ai_system import Pipeline, Data, BaseConfig, steps, factories

# 프로젝트의 기본 디렉토리 설정
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# 로거 설정 ('pybo'라는 이름의 로거를 생성)
logger = logging.getLogger('fastapi')

def process_image(image_path: str, selected_detectors: list) -> tuple:
    """
    이미지를 처리하여 얼굴을 탐지하고, 그 결과를 인코딩(임베딩)하는 함수입니다.

    Parameters
    ----------
    image_path : str
        처리할 이미지의 경로 (파일 경로).
    selected_detectors : list
        사용할 얼굴 탐지기 목록 (리스트).

    Returns
    -------
    output_image_path : str
        처리된 이미지가 저장된 경로 (파일 경로).
    encodings : list
        얼굴 특징 값을 인코딩한 결과값 리스트.
    """
    # 설정 파일에서 필요한 정보를 가져옵니다.
    config = BaseConfig.get_config()

    # 선택된 탐지기를 생성합니다.
    detectors = []
    for detector in selected_detectors:
        if detector == 'mtcnn':  # 'mtcnn' 탐지기는 사용하지 않음
            continue
        detectors.append(factories.FaceDetectorFactory.create(detector, config[f'{detector}']))

    # 얼굴 탐지 및 인코딩을 위한 파이프라인 생성
    pipeline = Pipeline()
    pipeline.add(steps.FaceDetector(detectors))  # 얼굴 탐지 단계
    pipeline.add(steps.FaceEncoder())  # 얼굴 인코딩 단계

    logging.info(f"이미지 처리 시작: {image_path}")

    # 데이터 객체를 생성하여 이미지 경로 설정
    data = Data(config, image_path)
    data.image_rgb  # 이미지의 RGB 값을 설정

    # 파이프라인 실행: 각 단계를 순차적으로 수행
    pipeline.run(data)

    logging.info(f"이미지 처리 완료: {image_path}")

    # 처리된 이미지 경로와 얼굴 인코딩 결과 반환
    output_image_path = data.output_image_path
    encodings = data.encodings
    return output_image_path, encodings

def compare_faces_ai(image1_path: str, image2_path: str, selected_detectors: list = ['yolo']) -> float:
    """
    두 이미지의 얼굴을 비교하여 유사도를 계산하는 함수입니다.

    Parameters
    ----------
    image1_path : str
        첫 번째 이미지의 경로 (파일 경로).
    image2_path : str
        두 번째 이미지의 경로 (파일 경로).
    selected_detectors : list, optional
        사용할 얼굴 탐지기 목록 (리스트, 기본값은 ['yolo']).

    Returns
    -------
    similarity_percentage : float
        두 얼굴 간의 유사도를 나타내는 값. 0에 가까울수록 유사함.
    """
    logger.info(f"첫 번째 이미지 인코딩 시작: {image1_path}")
    # 첫 번째 이미지 처리
    _, face_encoding1 = process_image(image1_path, selected_detectors)

    logger.info(f"두 번째 이미지 인코딩 시작: {image2_path}")
    # 두 번째 이미지 처리
    _, face_encoding2 = process_image(image2_path, selected_detectors)

    # 얼굴이 정확히 1개씩 있어야 비교 가능
    if len(face_encoding1) != 1:
        raise ValueError("첫 번째 사진에 얼굴이 1개가 아닙니다.")
    elif len(face_encoding2) != 1:
        raise ValueError("두 번째 사진에 얼굴이 1개가 아닙니다.")

    # 인코딩된 결과를 2차원 배열로 변환 (코사인 유사도 계산을 위해)
    face_encoding1 = np.array(face_encoding1[0]).reshape(1, -1)
    face_encoding2 = np.array(face_encoding2[0]).reshape(1, -1)

    logger.info(f"얼굴 유사도 계산 중: {image1_path}, {image2_path}")

    # 코사인 유사도 계산
    similarity_score = cosine_similarity(face_encoding1, face_encoding2)

    # 유사도 점수를 퍼센트로 변환 (0 ~ 100%)
    similarity_percentage = similarity_score[0][0] * 100

    # 유사도가 70% 이상일 때 0 ~ 100으로 변환
    if similarity_percentage >= 70:
        similarity_percentage = np.interp(similarity_percentage, [70, 100], [0, 100])

    return similarity_percentage


class DetectionConfig(BaseConfig):
    """
    탐지 설정을 관리하는 클래스입니다.
    YOLO 모델과 Django 디렉토리 및 결과 저장 경로를 설정합니다.
    """
    yolo_president_path = 'yolov8_l_president.pt'
    django_dir = BASE_DIR
    results_folder = os.path.join(django_dir, 'media', 'detection', 'a_image1')


def detect_president_ai(image_path: str, selected_detectors: list = ['yolo_president']) -> tuple:
    """
    이미지를 처리하고 얼굴을 탐지하여, 바운딩 박스를 이미지에 그려 저장하는 함수입니다.

    Parameters
    ----------
    image_path : str
        처리할 이미지의 경로 (파일 경로).
    selected_detectors : list, optional
        사용할 얼굴 탐지기 목록 (리스트, 기본값은 ['yolo_president']).

    Returns
    -------
    output_image_path : str
        처리된 이미지가 저장된 경로 (파일 경로).
    president_list : list
        탐지된 대통령의 이름 목록.
    """
    # 탐지 설정을 가져옵니다.
    config = DetectionConfig.get_config()

    # 선택된 탐지기를 생성합니다.
    detectors = []
    for detector in selected_detectors:
        if detector == 'mtcnn':  # 'mtcnn' 탐지기는 사용하지 않음
            continue
        detectors.append(factories.FaceDetectorFactory.create(detector, config[f'{detector}']))

    # 얼굴 탐지, 정보 그리기, 저장을 위한 파이프라인 생성
    pipeline = Pipeline()
    pipeline.add(steps.FaceDetector(detectors))  # 얼굴 탐지 단계
    pipeline.add(steps.InfoDrawer(thickness=5))  # 탐지 정보 그리기 단계
    pipeline.add(steps.Saver())  # 이미지 저장 단계

    logging.info(f"이미지 처리 시작: {image_path}")

    # 데이터 객체를 생성하고 이미지 경로 설정
    data = Data(config, image_path)

    # 파이프라인 실행: 각 단계를 순차적으로 수행
    pipeline.run(data)

    logging.info(f"이미지 처리 완료: {image_path}")

    # 처리된 이미지 경로와 대통령 이름 목록 반환
    output_image_path = data.output_image_path
    president_list = data.president_name_list

    return output_image_path, president_list
