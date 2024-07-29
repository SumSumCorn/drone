# 필요한 라이브러리들을 가져옵니다.
import cv2  # 컴퓨터 비전 라이브러리
from djitellopy import Tello  # DJI Tello 드론 제어 라이브러리
import streamlit as st  # 웹 앱 만들기 라이브러리
from object_detector import ObjectDetector  # 우리가 만든 물체 감지기
import time  # 시간 관련 기능을 위한 라이브러리
from playsound import playsound  # 소리 재생을 위한 라이브러리
import threading  # 여러 작업을 동시에 실행하기 위한 라이브러리

# Streamlit 페이지 설정
st.set_page_config(page_title="드론 유해동물 퇴치", layout="centered")
st.title("드론 유해동물 퇴치 시스템")

# 세션 상태 초기화 (앱에서 사용할 변수들을 설정합니다)
if "drone_connected" not in st.session_state:
    st.session_state.drone_connected = False  # 드론 연결 상태
if "drone" not in st.session_state:
    st.session_state.drone = None  # 드론 객체
if "webcam_connected" not in st.session_state:
    st.session_state.webcam_connected = False  # 웹캠 연결 상태
if "webcam" not in st.session_state:
    st.session_state.webcam = None  # 웹캠 객체
if "flying" not in st.session_state:
    st.session_state.flying = False  # 드론 비행 상태
if "bird_chase_mode" not in st.session_state:
    st.session_state.bird_chase_mode = False  # 새 추적 모드 상태
if "last_warning_time" not in st.session_state:
    st.session_state.last_warning_time = 0  # 마지막 경고음 재생 시간
if "patrol_mode" not in st.session_state:
    st.session_state.patrol_mode = False  # 정찰 모드 상태


# 경고음 재생 함수
def play_warning_sound():
    current_time = time.time()
    if current_time - st.session_state.last_warning_time > 5:  # 5초 간격으로 제한
        try:
            playsound("alarm.mp3")  # alarm.mp3 파일을 재생합니다
            st.session_state.last_warning_time = current_time
        except Exception as e:
            st.error(f"경고음 재생 중 오류 발생: {str(e)}")


# 드론 초기화 함수
@st.cache_resource
def init_drone():
    try:
        drone = Tello()  # Tello 드론 객체 생성
        drone.connect()  # 드론에 연결
        drone.streamon()  # 비디오 스트림 시작
        return drone
    except Exception as e:
        st.error(f"드론 연결 중 오류 발생: {str(e)}")
        return None


# 웹캠 초기화 함수
@st.cache_resource
def init_webcam():
    return cv2.VideoCapture(0)  # 기본 카메라(0번)를 사용하여 웹캠 초기화


# 물체 감지기 초기화 함수
@st.cache_resource
def init_detector():
    try:
        detector = ObjectDetector()  # ObjectDetector 클래스의 인스턴스 생성
        detector.initialize_object_detector()  # 물체 감지기 초기화
        return detector
    except Exception as e:
        st.error(f"객체 감지기 초기화 중 오류 발생: {str(e)}")
        return None


# 비디오 스트림 처리 함수 (Streamlit 프래그먼트로 실행)
@st.experimental_fragment(run_every=0.1)
def stream_video(video_source, detector):
    try:
        # 드론 또는 웹캠에서 프레임 가져오기
        if st.session_state.drone_connected:
            frame = video_source.get_frame_read().frame
        elif st.session_state.webcam_connected:
            ret, frame = video_source.read()
            if not ret:
                st.error("웹캠에서 프레임을 읽을 수 없습니다.")
                return
        else:
            st.error("비디오 소스가 연결되어 있지 않습니다.")
            return

        # 프레임 처리 및 물체 감지
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        detection_result = detector.detect_objects(frame_rgb)  # 물체 감지 실행
        processed_frame = detector.draw_detection_results(
            frame_rgb, detection_result
        )  # 감지 결과 그리기
        st.image(processed_frame, channels="BGR")  # Streamlit에 이미지 표시

        # 감지된 물체 정보 표시
        object_coords = detector.get_object_coordinates(frame_rgb, detection_result)
        for obj in object_coords:
            st.write(f"감지된 객체: {obj[0]}, 위치: (x={obj[1]}, y={obj[2]})")
    except Exception as e:
        st.error(f"비디오 스트리밍 중 오류 발생: {str(e)}")


# 새 추적 모드 함수
def chase_bird(drone, detector):
    drone.takeoff()  # 드론 이륙
    while True:
        try:
            frame = drone.get_frame_read().frame  # 드론에서 프레임 가져오기
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
            detection_result = detector.detect_objects(frame_rgb)  # 물체 감지

            bird_detected = False
            for detection in detection_result.detections:
                if (
                    detection.categories[0].category_name == "person"
                ):  # '사람'을 '새'로 가정
                    bird_detected = True
                    bbox = detection.bounding_box
                    frame_height, frame_width = frame.shape[:2]

                    # 새의 중심 좌표 계산
                    bird_center_x = bbox.origin_x + bbox.width / 2
                    bird_center_y = bbox.origin_y + bbox.height / 2

                    # 드론 제어를 위한 값 계산
                    left_right = int(
                        (bird_center_x - frame_width / 2) / (frame_width / 2) * 100
                    )
                    up_down = int(
                        (frame_height / 2 - bird_center_y) / (frame_height / 2) * 100
                    )
                    forward_backward = 0
                    yaw = 0

                    # 새의 크기에 따른 전진/후진 결정
                    bird_size = bbox.width * bbox.height
                    frame_size = frame_width * frame_height
                    size_ratio = bird_size / frame_size

                    if size_ratio < 0.5:
                        forward_backward = 30  # 새가 작으면 전진
                    # 새가 너무 가까우면 경고음 재생
                    # current_time = time.time()
                    # if (
                    #     current_time - st.session_state.last_warning_time > 5
                    # ):  # 5초 간격으로 제한
                    #     threading.Thread(
                    #         target=play_warning_sound, daemon=True
                    #     ).start()
                    # if size_ratio > 0.7:
                    #     # 새가 너무 가까우면 경고음 재생
                    #     current_time = time.time()
                    #     if (
                    #         current_time - st.session_state.last_warning_time > 5
                    #     ):  # 5초 간격으로 제한
                    #         threading.Thread(
                    #             target=play_warning_sound, daemon=True
                    #         ).start()

                    # 드론 제어 명령 전송
                    drone.send_rc_control(left_right, forward_backward, up_down, yaw)
                    break

            if not bird_detected:
                drone.send_rc_control(0, 0, 0, 0)  # 새가 없으면 정지

            # time.sleep(0.1)  # 제어 명령 간 짧은 대기 시간

        except Exception as e:
            st.error(f"새 추적 중 오류 발생: {str(e)}")
            break


# 정찰 모드 함수
def patrol_mode(drone):
    drone.takeoff()  # 드론 이륙
    for i in range(4):
        drone.move_forward(100)  # 앞으로 100cm 이동
        drone.rotate_clockwise(90)
    drone.land()  # 드론 착륙
    # 여기에 더 복잡한 정찰 로직을 추가할 수 있습니다


# 드론 동작 시뮬레이션 함수 (실제 드론이 없을 때 사용)
def simulate_drone_action(action):
    if action == "takeoff":
        st.session_state.flying = True
        st.success("드론이 이륙했습니다. (시뮬레이션)")
    elif action == "land":
        st.session_state.flying = False
        st.success("드론이 착륙했습니다. (시뮬레이션)")
    elif action == "emergency":
        st.session_state.flying = False
        st.warning("긴급 정지가 실행되었습니다. (시뮬레이션)")


# 메인 함수 (앱의 주요 로직)
def main():
    detector = init_detector()  # 물체 감지기 초기화

    # 드론 연결 버튼과 웹캠 연결 버튼을 나란히 배치
    col1, col2 = st.columns(2)

    with col1:
        if st.button("드론 연결"):
            if not st.session_state.drone_connected:
                st.session_state.drone = init_drone()
                if st.session_state.drone:
                    st.session_state.drone_connected = True
                    st.success("드론이 연결되었습니다.")
                else:
                    st.error("드론 연결 실패.")
            else:
                st.warning("드론이 이미 연결되어 있습니다.")

    with col2:
        if st.button("웹캠 연결"):
            if not st.session_state.drone_connected:
                st.session_state.webcam = init_webcam()
                if st.session_state.webcam.isOpened():
                    st.session_state.webcam_connected = True
                    st.success("웹캠이 연결되었습니다.")
                else:
                    st.error("웹캠 연결 실패.")
            else:
                st.warning(
                    "드론이 이미 연결되어 있습니다. 웹캠을 사용하려면 먼저 드론 연결을 해제하세요."
                )

    # 드론이 연결되었을 때 모드 선택 옵션 표시
    if st.session_state.drone_connected:
        mode = st.radio("모드 선택", ["일반", "새 추적", "정찰"])
        if mode == "새 추적":
            st.session_state.bird_chase_mode = True
            st.session_state.patrol_mode = False
            threading.Thread(
                target=chase_bird, args=(st.session_state.drone, detector), daemon=True
            ).start()
        elif mode == "정찰":
            st.session_state.bird_chase_mode = False
            st.session_state.patrol_mode = True
            st.session_state.drone.enable_mission_pads()
            threading.Thread(
                target=patrol_mode, args=(st.session_state.drone,), daemon=True
            ).start()
        else:
            st.session_state.bird_chase_mode = False
            st.session_state.patrol_mode = False
            st.session_state.drone.disable_mission_pads()

    # 비디오 소스 선택 (드론 또는 웹캠)
    video_source = (
        st.session_state.drone
        if st.session_state.drone_connected
        else st.session_state.webcam
    )

    if video_source:
        if st.session_state.drone_connected:
            st.write(f"배터리 잔량: {st.session_state.drone.get_battery()}%")
            if "drone_position" in st.session_state:
                st.write(f"드론 위치: {st.session_state.drone_position}")
        else:
            st.write("웹캠 사용 중")

        # 비디오 스트림 표시
        stream_video(video_source, detector)

        # 드론 제어 버튼
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("이륙"):
                if st.session_state.drone_connected:
                    st.session_state.drone.takeoff()
                else:
                    simulate_drone_action("takeoff")
        with col2:
            if st.button("착륙"):
                if st.session_state.drone_connected:
                    st.session_state.drone.land()
                else:
                    simulate_drone_action("land")
        with col3:
            if st.button("긴급 정지"):
                if st.session_state.drone_connected:
                    st.session_state.drone.emergency()
                else:
                    simulate_drone_action("emergency")


# 리소스 정리 함수
@st.cache_resource(ttl=None)
def cleanup_resources():
    if st.session_state.drone:
        st.session_state.drone.end()
    if st.session_state.webcam:
        st.session_state.webcam.release()
    st.session_state.drone_connected = False
    st.session_state.flying = False
    st.session_state.bird_chase_mode = False
    st.session_state.patrol_mode = False


# 앱 실행
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        cleanup_resources()
