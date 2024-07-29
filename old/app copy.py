import cv2
from djitellopy import Tello
import streamlit as st
from object_detector import ObjectDetector
from playsound import playsound
import threading

st.set_page_config(page_title="드론 유해동물 퇴치", layout="centered")
st.title("드론 유해동물 퇴치 시스템")

# 세션 상태 초기화
if "drone_connected" not in st.session_state:
    st.session_state.drone_connected = False
if "drone" not in st.session_state:
    st.session_state.drone = None
if "webcam" not in st.session_state:
    st.session_state.webcam = None
if "webcam_connected" not in st.session_state:
    st.session_state.webcam_connected = False
if "flying" not in st.session_state:
    st.session_state.flying = False


def play_warning_sound():
    try:
        playsound("alarm.mp3")
    except Exception as e:
        st.error(f"경고음 재생 중 오류 발생: {str(e)}")


@st.cache_resource
def init_drone():
    try:
        drone = Tello()
        drone.connect()
        drone.streamon()
        return drone
    except Exception as e:
        st.error(f"드론 연결 중 오류 발생: {str(e)}")
        return None


# @st.cache_resource
# def init_webcam():
#     return cv2.VideoCapture(0)
def init_webcam():
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        st.error("웹캠을 열 수 없습니다.")
        return None
    return webcam


@st.cache_resource
def init_detector():
    try:
        detector = ObjectDetector()
        detector.initialize_object_detector()
        return detector
    except Exception as e:
        st.error(f"객체 감지기 초기화 중 오류 발생: {str(e)}")
        return None


# @st.experimental_fragment(run_every=0.1)
# def stream_video(video_source, detector):
#     try:
#         if st.session_state.drone_connected:
#             frame = video_source.get_frame_read().frame
#         else:
#             ret, frame = video_source.read()
#             if not ret:
#                 st.error("웹캠에서 프레임을 읽을 수 없습니다.")
#                 return

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         detection_result = detector.detect_objects(frame_rgb)
#         processed_frame = detector.draw_detection_results(frame_rgb, detection_result)
#         st.image(processed_frame, channels="RGB")

#         object_coords = detector.get_object_coordinates(frame_rgb, detection_result)
#         for obj in object_coords:
#             st.write(f"감지된 객체: {obj[0]}, 위치: (x={obj[1]}, y={obj[2]})")
#     except Exception as e:
#         st.error(f"비디오 스트리밍 중 오류 발생: {str(e)}")


@st.experimental_fragment(run_every=0.1)
def stream_video(video_source, detector):
    try:
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

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect_objects(frame_rgb)
        processed_frame = detector.draw_detection_results(frame_rgb, detection_result)
        st.image(processed_frame, channels="RGB")

        object_coords = detector.get_object_coordinates(frame_rgb, detection_result)
        for obj in object_coords:
            st.write(f"감지된 객체: {obj[0]}, 위치: (x={obj[1]}, y={obj[2]})")
    except Exception as e:
        st.error(f"비디오 스트리밍 중 오류 발생: {str(e)}")


@st.experimental_fragment(run_every=0.1)
def chase_bird(detector):
    if not st.session_state.bird_chase_mode or not st.session_state.drone_connected:
        return

    try:
        frame = st.session_state.drone.get_frame_read().frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detection_result = detector.detect_objects(frame_rgb)

        bird_detected = False
        for detection in detection_result.detections:
            if detection.categories[0].category_name == "bird":
                bird_detected = True
                bbox = detection.bounding_box
                frame_height, frame_width = frame.shape[:2]

                # 새의 중심 좌표 계산
                bird_center_x = bbox.origin_x + bbox.width / 2
                bird_center_y = bbox.origin_y + bbox.height / 2

                # 드론 이동 명령 계산
                left_right = int(
                    (bird_center_x - frame_width / 2) / (frame_width / 2) * 100
                )
                up_down = int(
                    (frame_height / 2 - bird_center_y) / (frame_height / 2) * 100
                )
                forward_backward = 0
                yaw = 0

                # 새와의 거리 추정 (bbox 크기 기반)
                bird_size = bbox.width * bbox.height
                frame_size = frame_width * frame_height
                size_ratio = bird_size / frame_size

                if size_ratio < 0.1:  # 새가 화면의 10% 미만을 차지할 때
                    forward_backward = 30
                if size_ratio > 0.3:  # 새가 화면의 30% 이상을 차지할 때
                    # 경고음 재생 (별도의 스레드에서 실행)
                    threading.Thread(target=play_warning_sound, daemon=True).start()
                    st.warning("경고음 재생 중")

                # rc_control 함수 사용
                st.session_state.drone.send_rc_control(
                    left_right, forward_backward, up_down, yaw
                )

                st.write("새 감지됨: 추적 중")
                break  # 첫 번째 감지된 새만 처리

        if not bird_detected:
            # 새가 감지되지 않으면 제자리 비행
            st.session_state.drone.send_rc_control(0, 0, 0, 0)

    except Exception as e:
        st.error(f"새 추적 중 오류 발생: {str(e)}")


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


def main():
    detector = init_detector()

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
                if st.session_state.webcam and st.session_state.webcam.isOpened():
                    st.session_state.webcam_connected = True
                    st.success("웹캠이 연결되었습니다.")
                else:
                    st.error("웹캠 연결 실패.")
            else:
                st.warning(
                    "드론이 이미 연결되어 있습니다. 웹캠을 사용하려면 먼저 드론 연결을 해제하세요."
                )

    if st.session_state.drone_connected:
        if st.button(
            "새 쫓기 모드 " + ("끄기" if st.session_state.bird_chase_mode else "켜기")
        ):
            st.session_state.bird_chase_mode = not st.session_state.bird_chase_mode

    video_source = (
        st.session_state.drone
        if st.session_state.drone_connected
        else st.session_state.webcam
    )

    if video_source:
        if st.session_state.drone_connected:
            st.write(f"배터리 잔량: {st.session_state.drone.get_battery()}%")
        else:
            st.write("웹캠 사용 중")

        stream_video(video_source, detector)

        if st.session_state.bird_chase_mode and st.session_state.drone_connected:
            chase_bird(detector)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("이륙", use_container_width=True):
                if st.session_state.drone_connected:
                    st.session_state.drone.takeoff()
                else:
                    simulate_drone_action("takeoff")
        with col2:
            if st.button("착륙", use_container_width=True):
                if st.session_state.drone_connected:
                    st.session_state.drone.land()
                else:
                    simulate_drone_action("land")
        with col3:
            if st.button("긴급 정지", use_container_width=True):
                if st.session_state.drone_connected:
                    st.session_state.drone.emergency()
                else:
                    simulate_drone_action("emergency")


# cleanup_resources 함수 수정
@st.cache_resource(ttl=None)
def cleanup_resources():
    if st.session_state.drone:
        st.session_state.drone.end()
    if st.session_state.webcam:
        st.session_state.webcam.release()
    st.session_state.drone_connected = False
    st.session_state.webcam_connected = False
    st.session_state.flying = False


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        cleanup_resources()
