import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

class ObjectDetector:
    def __init__(self, model_path="model/efficientdet_lite0.tflite"):
        self.model_path = model_path
        self.detector = None

    def initialize_object_detector(self):
        """
        MediaPipe 물체 감지기를 초기화합니다.
        """
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=0.5,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.detector = vision.ObjectDetector.create_from_options(options)

    def detect_objects(self, frame):
        """
        주어진 프레임에서 물체를 감지합니다.
        매개변수:
        - frame: 비디오 프레임 (numpy 배열)
        반환값: 감지된 물체 목록
        """
        if self.detector is None:
            self.initialize_object_detector()

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = self.detector.detect(mp_image)

        return detection_result

    def draw_detection_results(self, image, detection_result):
        """
        감지 결과를 프레임에 그립니다.
        매개변수:
        - image: 입력 RGB 이미지.
        - detection_result: MediaPipe의 감지 결과
        반환값: 물체가 표시된 이미지
        """
        for detection in detection_result.detections:
            # 경계 상자 그리기
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(image, start_point, end_point, TEXT_COLOR, 3)

            # 레이블과 점수 그리기
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            result_text = f"{category_name} ({probability})"
            text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return image

    def process_video_stream(self, video_frame):
        """
        비디오 스트림을 처리하여 물체를 감지하고 결과를 표시합니다.
        매개변수:
        - video_frame: 비디오 프레임 (numpy 배열)
        반환값: 물체가 감지된 프레임
        """
        detection_result = self.detect_objects(video_frame)
        processed_frame = self.draw_detection_results(video_frame, detection_result)
        return processed_frame

    def get_object_coordinates(self, frame, detection_result):
        """
        감지된 물체의 좌표를 반환합니다.
        매개변수:
        - frame: 비디오 프레임 (numpy 배열)
        - detection_result: MediaPipe의 감지 결과
        반환값: 물체 좌표 리스트 [(label, x, y, w, h), ...]
        """
        object_coords = []
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            category = detection.categories[0]
            label = category.category_name
            object_coords.append(
                (label, bbox.origin_x, bbox.origin_y, bbox.width, bbox.height)
            )
        return object_coords

# 사용 예시
if __name__ == "__main__":
    detector = ObjectDetector()
    detector.initialize_object_detector()

    # 웹캠에서 비디오 스트림 가져오기 (테스트용)
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("비디오 스트림을 읽을 수 없습니다.")
            break

        # 물체 감지 및 결과 표시
        processed_frame = detector.process_video_stream(frame)

        # 감지된 물체 좌표 가져오기
        detections = detector.detect_objects(frame)
        object_coords = detector.get_object_coordinates(frame, detections)

        # 좌표 출력 (테스트용)
        for obj in object_coords:
            print(f"Object: {obj[0]}, Coordinates: (x={obj[1]}, y={obj[2]}, w={obj[3]}, h={obj[4]})")

        cv2.imshow("Object Detection", processed_frame)
        if cv2.waitKey(5) & 0xFF == 27:  # ESC 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()