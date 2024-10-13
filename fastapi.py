import os
import platform
import random
import pandas as pd
import base64
from fastapi import FastAPI, File, UploadFile
from ai import detect_president_ai, compare_faces_ai

# FastAPI 애플리케이션 생성
app = FastAPI()

# 미국 대통령에 대한 설명을 담은 데이터
president_info= {
    '존슨':"이 사진 속 인물은 린든 B. 존슨(Lyndon B. Johnson)입니다. 린든 B. 존슨은 미국의 36대 대통령으로, 1963년부터 1969년까지 재임했습니다. 그는 대사회 프로그램과 민권법을 통해 사회 복지와 인권을 강화했으나, 베트남 전쟁으로 인해 큰 논란에 휘말렸습니다. 그의 정책은 오늘날에도 여전히 중요한 논의의 주제가 되고 있습니다",
    "닉슨":"이 사진 속 인물은 리처트 닉슨(Richard Nixon)입니다. 리처드 닉슨은 미국의 37대 대통령으로, 1969년부터 1974년까지 재임했습니다. 그는 중국과의 외교 관계 정상화와 워터게이트 스캔들로 인해 사임하는 등 극명한 대조를 이루는 정치 경력을 가졌습니다. 또한, 환경 보호법과 같은 중요한 국내 정책도 추진했습니다.",
    "포드":"이 사진 속 인물은 제럴드 포드(Gerald Ford)입니다. 제럴드 포드는 미국의 38대 대통령으로, 1974년부터 1977년까지 재임했습니다. 그는 리처드 닉슨의 사임 후 부통령으로 취임했으며, 경제적 어려움과 높은 인플레이션에 대응하기 위해 다양한 캠페인을 추진했습니다. 닉슨을 사면한 결정은 큰 논란을 일으켰고, 그의 재임 기간은 정치적으로 힘든 시기로 기억됩니다.",
    "카터":"이 사진 속 인물은 지미 카터(Jimmy Carter)입니다.지미 카터는 미국의 39대 대통령으로, 1977년부터 1981년까지 재임했습니다. 그는 인권을 강조하고 에너지 위기 대응에 노력했으며, 캠프 데이비드 협정을 통해 중동 평화에 기여했습니다. 퇴임 후에도 인권과 사회 봉사 활동으로 많은 영향을 미쳤습니다.",
    "레이건":"이 사진 속 인물은 로널드 레이건(Ronald Reagan)입니다.로널드 레이건은 미국의 40대 대통령으로, 1981년부터 1989년까지 재임했습니다. 그는 공급 측 경제학을 통해 경제 성장과 세금 감면을 추진하고, 냉전 종식에 기여했습니다. 또한, 군사력 증강과 사회 복지 프로그램 축소를 통해 보수주의 정책을 강화했습니다.",
    "아빠 부시":"이 사진 속 인물은 조지 H.W. 부시((George H.W. Bush)입니다. 조지 H.W. 부시는 미국의 41대 대통령으로, 1989년부터 1993년까지 재임했습니다. 그는 냉전 종식과 걸프 전쟁에서의 군사 작전으로 국제 정치에서 중요한 역할을 했으나, 재임 중 경제 침체와 세금 인상 문제로 비판을 받았습니다. 또한, 환경 보호와 장애인 권리 증진을 위한 법안을 추진하며 사회 정책에도 기여했습니다.",
    "클린턴":"이 사진 속 인물은 빌 클린턴(Bill Clinton)입니다빌 클린턴은 미국의 42대 대통령으로, 1993년부터 2001년까지 재임했습니다. 그는 재임 중 경제 성장을 이끌고 여러 사회 정책을 추진했으나, 모니카 르윈스키 스캔들로 인해 탄핵 절차를 겪었습니다. 클린턴 대통령은 경제적 성과와 정치적 논란이 얽힌 복잡한 평가를 받고 있습니다." ,
    "아들 부시":"이 사진 속 인물은 조지 W. 부시(George W. Bush)입니다,조지 W. 부시는 미국의 43대 대통령으로, 2001년부터 2009년까지 재임했습니다. 그는 9/11 테러 이후 아프가니스탄과 이라크 전쟁을 주도하며 테러와의 전쟁을 선언했으나, 2008년 금융 위기로 인해 경제 정책에 대한 비판을 받았습니다. 교육 개혁과 보건 정책에도 관심을 기울였지만, 그의 재임 기간은 전쟁과 경제 문제로 논란이 많았습니다.",
    '오바마': "이 사진 속 인물은 오바마 (Barack Obama)입니다.오바마는 미국의 44대 대통령으로, 2009년부터 2017년까지 재임했습니다. 그는 미국 역사상 첫 아프리카계 미국인 대통령이며, 재임 중 건강보험 개혁, 재정위기 대응, 외교 정책 변화 등의 여러 중요한 정책을 추진했습니다. 퇴임 후에도 사회적 이슈에 대한 목소리를 내고 있으며, 많은 사람들에게 영감을 주고 있습니다. ",
    "트럼프":"이 사진 속 인물은 도널드 트럼프(Donald Trump)입니다. 그는 미국의 제45대 대통령으로 2017년부터 2021년까지 재임했으며, 정치인이기 이전에는 부동산 개발업자이자 TV 방송인으로도 유명했습니다. 트럼프는 2016년 대통령 선거에서 공화당 후보로 출마해 승리했으며, 재임 중에는 ‘미국 우선주의’를 내세워 보호무역, 이민 제한, 세금 감면 등의 정책을 추진했습니다.",
    "바이든":"이 사진 속 인물은 조 바이든(Joe Biden)입니다.조 바이든은 미국의 46대 대통령으로, 2021년 1월 20일부터 재임 중입니다. 그는 COVID-19 대응과 경제 회복을 위한 법안을 추진하며 기후 변화 및 인종 평등과 같은 진보적 사회 정책을 지향하고 있습니다. 또한, 동맹국과의 관계를 강화하고 중국과의 경쟁에서 미국의 입장을 확립하려고 노력하고 있습니다."
}

# 이미지 파일을 저장할 폴더 생성
UPLOAD_AI_IMAGE_FOLDER = 'detection_image'  # AI 분석 이미지 저장 경로
UPLOAD_SIMILARITY_IMAGE_FOLDER = 'similarity_image'  # 유사도 분석 이미지 저장 경로
os.makedirs(UPLOAD_AI_IMAGE_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_SIMILARITY_IMAGE_FOLDER, exist_ok=True)

async def save_uploaded_file(upload_folder: str, file: UploadFile) -> str:
    """
    클라이언트로부터 업로드된 파일을 지정된 폴더에 비동기적으로 저장합니다.

    Parameters
    ----------
    upload_folder : str
        파일을 저장할 경로를 나타내는 문자열입니다. 이 경로는 미리 생성된 폴더입니다.
    
    file : UploadFile
        FastAPI에서 제공하는 UploadFile 객체로, 클라이언트가 업로드한 파일을 담고 있습니다.

    Returns
    -------
    str
        저장된 파일의 경로를 문자열로 반환합니다.

    Notes
    -----
    - FastAPI의 UploadFile은 비동기 파일 작업을 지원하며, 파일을 읽고 저장하는 작업은 비동기로 수행됩니다.
    - 파일은 지정된 폴더 내에 저장되며, 해당 경로를 반환하여 후속 작업에 사용됩니다.
    """
    file_path = os.path.join(upload_folder, file.filename)
    
    # FastAPI의 UploadFile.read()를 사용하여 파일을 비동기적으로 읽음
    content = await file.read()

    # 일반적인 open()을 사용해 파일 저장
    with open(file_path, "wb") as image_file:
        image_file.write(content)
    
    return file_path

@app.post("/detect_president/")
async def detect_president_api(file: UploadFile = File(...)) -> dict:
    """
    업로드된 이미지를 AI로 분석하여 해당 이미지 속 인물을 식별하고 결과를 반환합니다.

    Parameters
    ----------
    file : UploadFile
        클라이언트가 업로드한 이미지 파일. 이 파일은 FastAPI의 File()을 통해 전달받습니다.

    Returns
    -------
    dict
        처리된 이미지 속 인물에 대한 설명과 base64로 인코딩된 이미지 데이터를 담은 사전 형식의 결과를 반환합니다.

    Raises
    ------
    Exception
        이미지 처리 중 오류가 발생할 경우 예외 메시지가 반환됩니다.

    Notes
    -----
    - 이 함수는 먼저 업로드된 파일을 서버에 저장하고, 그 파일을 AI를 통해 분석합니다.
    - 분석된 결과는 인물에 대한 설명과 AI가 처리한 이미지를 base64로 인코딩한 데이터로 반환됩니다.
    """
    # 파일을 지정된 경로에 저장
    file_path = await save_uploaded_file(UPLOAD_AI_IMAGE_FOLDER, file)
    try:
        # AI 분석을 통해 얼굴 탐지 및 인물 식별
        result_image_path, president_list = detect_president_ai(file_path)

        # 첫 번째로 탐지된 인물에 대한 설명 가져오기
        president_name = president_list[0]
        president_description = president_info.get(president_name, "알 수 없는 인물입니다.")

        # 결과 이미지 파일을 읽고 base64로 인코딩
        with open(result_image_path, "rb") as result_image_file:
            result_image_data = base64.b64encode(result_image_file.read()).decode('utf-8')

        return {
            'message': president_description,
            'base64_image': result_image_data,
            'image_path': result_image_path
        }
    except Exception as e:
        return {"message": f"이미지 처리 중 오류가 발생했습니다: {str(e)}"}
    finally:
        # 파일 삭제
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(result_image_path):
            os.remove(result_image_path)

@app.post("/compare_similarity/")
async def compare_similarity_api(file1: UploadFile = File(...), file2: UploadFile = File(...)) -> dict:
    """
    두 이미지를 AI로 분석하여 얼굴 유사도를 계산하고 결과를 반환합니다.

    Parameters
    ----------
    file1 : UploadFile
        첫 번째 이미지 파일.
    file2 : UploadFile
        두 번째 이미지 파일.

    Returns
    -------
    dict
        두 이미지의 유사도를 나타내는 메시지를 포함한 사전 형식의 결과를 반환합니다.

    Raises
    ------
    Exception
        AI 처리 중 오류가 발생할 경우 예외 메시지가 반환됩니다.

    Notes
    -----
    - 두 이미지를 서버에 저장한 후, 유사도를 AI 모델로 분석하여 결과를 반환합니다.
    - 유사도에 따른 적절한 설명 메시지도 함께 반환됩니다.
    """
    # 파일을 저장
    file1_path = await save_uploaded_file(UPLOAD_SIMILARITY_IMAGE_FOLDER, file1)
    file2_path = await save_uploaded_file(UPLOAD_SIMILARITY_IMAGE_FOLDER, file2)
    try:
        # AI를 이용해 두 얼굴 간의 유사도 계산
        similarity_score = compare_faces_ai(file1_path, file2_path)

        # 유사도에 따른 설명 메시지 생성
        similarity_message = f"두 사진의 유사도는 {similarity_score:.2f}%입니다.\n{get_similarity_message(similarity_score)}"
        return {'result': similarity_message}

    except Exception as e:
        return {'result': f"AI 처리 중 오류가 발생했습니다: {str(e)}"}
    
    finally:
        # 작업 후 파일 삭제
        if os.path.exists(file1_path):
            os.remove(file1_path)
        if os.path.exists(file2_path):
            os.remove(file2_path)

def get_similarity_message(similarity: float) -> str:
    """
    두 이미지의 유사도 값에 따라 적절한 메시지를 반환합니다.
    
    로컬 호스트가 아닌 윈도우 환경인 경우와 서버 환경을 구분하여 처리합니다.
    윈도우 환경일 경우 '정상작동 중 입니다' 메시지를 반환하고, 
    서버 환경에서는 CSV 파일을 읽어 랜덤한 메시지를 반환합니다.

    Parameters
    ----------
    similarity : float
        두 이미지 간의 유사도 값.

    Returns
    -------
    str
        유사도에 따른 설명 메시지.
    """
    # 유사도를 정수로 변환
    similarity = int(similarity)

    # 유사도에 따라 선택할 컬럼 결정
    if similarity < 30:
        column_index = 0
    elif 30 <= similarity < 60:
        column_index = 1
    elif 60 <= similarity < 80:
        column_index = 2
    else:
        column_index = 3

    # 현재 운영체제가 윈도우인지 확인
    if platform.system() == "Windows":
        message = '윈도우 환경에서 정상작동 중 입니다'
    else:
        # 실제 서버 환경에서는 CSV 파일을 읽어 랜덤한 메시지 반환
        try:
            data = pd.read_csv("similarity_text.csv", encoding='cp949')
            random_row = random.randint(0, len(data) - 1)
            message = data.iloc[random_row, column_index]
        except Exception as e:
            message = f"메시지를 불러오는 중 오류가 발생했습니다: {e}"

    return message

# FastAPI 애플리케이션 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8007)