# aiot_2024_robot

aiot robot

## 구글 슬라이드 링크

[클릭](https://docs.google.com/presentation/d/1LHXvZnWKbSJLpbeop8mL-gTkLUKbxCPuk6moOfNLY-s/edit#slide=id.p)

## gazebo 환경 변수

```bash
export SVGA_VGPU10=0
```

---

## 2024_9_19

---

- python 설치
- python version 확인
- python VsCode 사용법
- 1장
  - 키워드, 식별자, 변수, 자료형
  - type()
  - 파이썬에서의 변수 클래스의 객체 관계
  - 연산자, 연산자 오버라이딩
- 2 교시
  - print, 함수 읽는 법, sep, end
- 3 교시
  - str class, indexing, slicing, len
- 4 교시
  - 복합대입연산자
  - input
  - str format, f-string
- 5 교시
  - 불리언 자료형
  - if condition
- 6 교시
  - datetime module
- 7 교시
  - 예시 -- 계절 구하기, 홀수 짝수 구하기, 학점

---

## 2024_9_20

---

- 1교시
  - 리스트
  - 책 교부
- 2교시
  - range for 문
- 3교시
  - dictionary, for 문
- 4교시
  - while, list method, dictionary method
  - list comprehension, enumerate, 삼항연산자
- 5교시
  - 함수 정의 및 기본 구조
- 6교시
  - 함수의 인자 (postional-argument, default-argument, keyword-argument, variable-length-argument, keyword-variable-length-argument)
  - 함수의 반환값(return, 및 tuple)
- 7교시
  - 예외 처리 try, except, else, finally
  - 사용자 정의 예외처리, raise

---

## 2024_09_23

---

- 1교시
  - 클래스 개념
- 2교시
  - data 로 처리
  - 간단한 클래스 만들기
  - dataclass 데코레이터
- 3교시
  - method 추가
  - special method 추가 (비교 연산자 및 str, expr)
- 4교시
  - 클래스 변수 활용
  - 클래스메소드 classmethod 데코레이터
  - 다중 상속 및 mro
- 5교시
  - private 설정
  - property 데코레이터 getter, setter
- 6교시
  - tuple exchange
  - 재귀함수 만들기
  - lru_cache 데코레이터
- 7교시
  - 람다함수
  - 파일 처리
  - 제너레이터
  - 램덤 모듈

---

## 2024_09_24

---

- 1교시
  - 복습
  - module 개념 실습
- 2교시
  - package 개념 실습
  - import 및 __init__py 실습
- 3교시

---

## 2024_10_14

---

- 1교시
  - OpenCV 설치 ( cpp 설치, python 설치)
  - OpenCV 기본 사용법
  - make, cmake 사용법
- 2교시
  - 기본 함수
  - imread, imshow, waitKey, destroyAllWindows
  - VideoCapture, VideoWriter
- 3교시
  - 기본 클래스
  - Point_ 클래스
  - Size_ 클래스
- 4교시
  - Rect_ 클래스
  - Scalar_ 클래스
  - Mat_ 클래스
- 5교시
  - draw 함수
  - line, rectangle, circle, ellipse, putText
  - freetype 사용법 (한글폰트)
- 6교시
  - python과 c++의 차이점
- 7교시
  - python draw 함수 실습

---

## 2024_10_15

---

- 1교시
  - 복습
  - 밝기 조절 ( + , add 함수)
  - saturate_cast
- 2교시
  - waitKeyEx 함수
  - 마우스 콜백 함수
- 3교시
  - python 마우스 콜백 함수
  - 대비 함수 (histogram, histogram equalization, stretching)
  - bitwise 연산
- 4교시
  - blur 함수
  - gaussian blur 함수
  - median blur 함수
- 5교시
  - warpAffine 함수
  - perspective transform
  - perspective transform 실습
- 6교시
  - 미분 필터
  - canny edge detection
- 7교시
  - hough line transform

---

## 2024_10_16

---

- 1교시
  - houghlineP 실습
  - trackbar 실습
- 2교시
  - color space 변환
  - inrange 함수
- 3교시
  - 이진화 함수 threshold, adaptiveThreshold
  - 모폴로지 연산
- 4교시
  - 템플릿 매칭
  - 캐스캐이드 검출
  - Hog 알고리즘
- 5교시
  - QR code 실습 ( cpp, python) cpp 코드 오브젝트 링크 오류
  - AruCo 실습
- 6교시
  - OpenCV 머신러닝
- 7교시
  - 필기체 인식 0~9 knn

---

## 2024_10_17

---

- 1교시
  - 필기체 인식 0~9 knn 2
- 2교시
  - OpenCV 딥러닝
  - 필기체 인시 0~9 cnn
  - 이미지 분류
- 3교시
  - ROS2 개념
- 4교시
  - ROS2 설치
- 5교시
  - ROS2 cli 실습
    - ros2 run, ros2 launch, ros2 topic, ros2 node, ros2 param, ros2 service, ros2 action
- 6교시
  - ROS2 rqt 실습
    - rqt_graph, rqt_plot, rqt_image_view, rqt_console, rqt_logger_level
- 7교시
  - pkg 만들기
    - ro2 pkg create
  - node 작성 python

---

## 2024_10_18

---

- 1교시
  - node 작성 기본 코드
  - rclpy.init, rclpy.spin
  - Node 클래스
- 2교시
  - alias 설정
  - easyinstall deprecated 에러
    - pip3 install setuptools==58.2.0
- 3교시
  - publisher 만들기
  - class 구조화 하기
- 4교시
  - subscription 코드 만들기
- 5교시
  - QoS 코드 설정
  - 시간 인터페이스 Header 사용하기
- 6교시
  - [과제]homework 패키지 만들기

---

## 2024_10_21

---

- 1교시
  - ros2 cpp 패키지 만들기 simple_ros_cpp
- 2교시
  - CMakeLists.txt 작성
  - .vscode/c_cpp_properties.json 수정
  - ros2 cpp publisher 만들기
- 3교시
  - cpp publisher 콜백함수를 lambda 함수로 만들기
  - class 구조화 하기.
- 4교시
  - cpp 분할 컴파일 및 include 추가
  - launch 파일 작성 및 적용(python cpp)
  - cout 대신 RCLCPP_INFO 사용하기
  - print 대신 self.get_logger().info 사용하기
- 5교시
  - ros2 cpp subscriber 만들기
  - 외부 라이브러리를 ros2 에서 사용하기 (opencv 글자 표시)
- 6교시
  - moveTurtle.py 기본 코드 작성
  - 사각형 그리기 코드
- 7교시
  - [과제]cpp 로 같은 코드 작성

---

## 2024_10_22

---

- 1교시
  - cpp 로 moveTurtle.cpp 작성 (simple_ros_cpp)
- 2교시
  - interface 설명 (topic, service, action) 차이
- 3교시
  - python service server 작성, service client 작성
  - 동기방식의 service 에서 비동기 방식으로 코드 작성하기 call_async
- 4교시
  - cpp service server 작성
- 5교시
  - cpp serivce client 작성
- 6교시
  - user interce 작성 topic UserInt class (user_interface 패키지)
    - package.xml, CMakeLists.txt 수정
  - user_int_pub 노드 작성 (simple_ros 패키지)
- 7교시

---

## 2024_10_23

---

- 1교시
  - 파라미터 적용 노드 작성
- 2교시:
  - 런치 파일 작성
  - 런치 파일에서 파라미터 사용
  - cli 에서 파라미터 파일(yaml) 적용
  - namespace 적용
- 4교시
  - 런치 파일로 터틀심노드 사용
- 5교시
  - action interface  추가( user_interface )
  - action server 작성(Fibonacci 코드)
- 6교시
  - action client 작성(Fibonacci 코드)
- 7교시
  - action client 작성(python type hint 추가)

---

## 2024_10_24

---

- 1교시
  - simple_parameter2 노드에서 simple_parameter 노드 의 파라미터 변경하기 (service 코드)
- 2교시
  - 파라미터 추가 설명 : 런치에서 여러노드의 파라미터 관리
  - action_server python 작성 (fibonacci)
- 3교시
  - cpp 파라미터 노드 작성
  - turtlesim 을 이용한 파라미터 노드 작성
  - 런치에서 파라미터 파일 적용하기
  - 코드, 런치, 실행문에서 파라미터 적용의 순서
- 4교시
  - namespace 설명
  - namespace 를 적용해서 turtlesim 노드 제어(2개의 터틀 제어)
  - 인터페이스 작성 fibonacci.action
- 5교시
  - 런치 파일 작성
  - action_client python 작성
- 6교시
  - cpp 런치 파일 작성 (python action_server, cpp action_client)
- 7교시
  - action_client cpp 작성 (fibonacci)

---

## 2024_10_25

---

- arithmetic 패키지 작성
- 1교시
  - 패키지 생성 arith
  - python 노드 작성 argument
  - python 노드 작성 calculator
- 2교시
  - calculator 노드 에 서비스 서버 추가
  - operator 노드 작성 ( 서비스 클라이언트)
- 3교시
  - calculator 노드 에 액션 서버 추가
  - checker 노드 작성 ( 액션 클라이언트)
- 4교시
  - calculator 노드에 멀티쓰레드 설정 추가
  - 런치 작성
- 5교시
  - cpp 패키지 생성 arith_cpp
  - cpp 노드 작성 argument
- 6교시
  - cpp 노드 작성 calculator
- 7교시
  - 런치 파일 작성

---

## 2024_10_28

---

- 1교시
  - 3부 심화 프로그래밍 로깅
  - 로깅 환경 변수 설정
  - 파이썬 로깅 노드 작성
- 2교시
  - CPP 로깅 노드 작성
- 3교시
  - 사용자 정의 cli 명령어 작성
  - 패키지 생성 ros2env
- 4교시
  - 사용자 정의 cli 명령어 작성
- 5교시
  - 터틀봇3 설명
    - 특징
    - 데이터계통, 전력계통
- 6교시
  - 패키지 설치
    - 가제보, 카토그래퍼, 네비게이션
    - 터틀봇3 패키지 설치
    - 터틀봇3 wifi 설정, ros2 domain 설정
- 7교시
  - 터틀봇3 제어 체크[실습]

---

## 2024_10_29

---

- 1교시
  - IPC 설명
  - cpp 노드 작성 two node pipeline.cpp
- 2교시
  - cpp 노드 작성 cyclic pipeline.cpp
- 3교시
  - cpp 노드 작성 image pipeline.cpp - 동영상 파일로 작동할 수 있게 수정
- 4교시
  - QOS 설정 ( qos_profile, history, depth, reliability, durability) 복습
  - deadline 예제 코드 deadline.py
- 5교시
  - deadline 예제 코드
- 6교시
  - 터틀봇 VsCode remote 연결
  - move_turtle 패키지 생성(foxy python)
- 7교시
  - circle 노드 작성 ( 터틀봇 cmd_vel 제어)
  - retancle 노드 작성 ( 터틀봇 cmd_vel 발행, odom 구독)

---

## 2024_10_30

---

- 1교시
  - 복습
  - lifespan QoS 설정 노드 작성
- 2교시
  - liveliness QoS 설정 노드 작성
- 3교시
  - component 원리, shared object 설명
  - component talker 노드 작성
- 4교시
  - component manager 로 노드 로드 실습
  - component listener 노드 작성
- 5교시
  - gazebo 설명
  - gazebo use_sim_time 실습
- 6교시
  - gazebo 에 turtlebot3 모델 불러오기.
  - 패키지 생성(move_turtle : humble)
  - 원 그리기 노드 적용
- 7교시
  - 모델을 작동하는데 필요한 state_publisher, tf 설명.
  - ros2 launch turtlebot3_gazebo empty_world.launch.py 로 실행.
  - rviz2 실습
  - 사각형 그리기 노드 디버깅( gazebo 시뮬레이션 이용)

---

## 2024_11_01

---

- 1교시
  - 복습
  - gazebo tf 와 turtlebot3_gazebo launch 파일 분석
  - component manager 설명
- 2교시
  - Custom Executable 실습[제한 사항: 헤더 파일이 없으면 불가능]
  - 보통 컴포넌트는 헤더 파일이 없음
  - Launch 에서 컴포넌트 매니저 사용 실습 (composition_demo.launch.py)
- 3교시
  - 플러그인 rqt 실습
  - rqt_example 작성 (rqt_example_widget 과 rqt_example ui 는 복사 해서 사용)
- 4교시
  - 빌드 및 rqt 플러그인 설치 실행
  - 인식 오류가 있어서 ~/.config/ros.org/rqt_gui.ini 파일을 수정
  - 파이썬 파일 강제 실행 python3 /scripts/rqt_example --force-discover
- 5교시
  - rqt 플러그인 설정 및 실습
- 6교시
  - 시뮬레이션으로 터틀봇3 사각형 그리기 실습
  - 터틀봇3 사각형 그리기 완성

---

## 2024_11_04

---

- 1교시
  - 복습
  - lifecycle 설명
  - lifecycle talker 노드 작성
- 2교시
  - lifecycle talker 노드 작성
  - ros2 lifecycle list, get, set, transition 명령어 사용
- 3교시
  - error 상황에서 노드 재시작 설정 (launch)
  - lifecycle listener 노드 작성
- 4교시
  - tf2 설명
  - turtle_tf_py 노드 설치 및 실행
  - static_tf2_broadcaster 노드 작성
  - cli 명령어로 ros2 run tf2_ros static_tf2_broadcaster 실행
- 5교시
  - tf2 launch 파일 작성 ( 외부 런치 파일 실행(rviz2, turtle_tf_py, etc node))
- 6교시
  - dynamic tf2 broadcaster 노드 작성
  - turtle1을 움직여서 rviz로 확인
- 7교시
  - tf2 listener 노드 작성

---

## 2024_11_05

---

- 1교시
  - 복습
  - sros2 설명, ros2 security create_key key_box
- 2교시
  - sros2 key 생성 및 사용
  - sros2 변수 적용시 ros2 가 실행 안되는 현상이 있어서 서치후 다시 시도하기로 함.
- 3교시
  - turtlebot3 의 tf 발행 robotis 깃허브 분석
  - tf 발행 위치 확인 urdf, turtlebot3_node
- 4교시
  - gazebo 설정
- 5교시
  - laserscan 데이터 분석 및 내부 데이터 업데이트 형식 설정 (라이다 센서)
- 6교시
  - move_turtle 의 follow_wall 노드 작성 (라이다 센서 활용)
- 7교시
  - follow_wall 노드 실습

---

## 2024_11_06

---

- 1교시
  - 복습
  - urdf 파일 작성 myfirtst.urdf
- 2교시
  - urdf_launch 패키지 설치 (sudo apt install ros-humble-urdf-launch)
  - move_turtle 에 display.launch.py 작성
  - rviz config 파일 작성
- 3교시
  - multipleshapes.urdf 작성 (link 여러개 추가)
  - origins.urdf 작성 origin 설정( 왼발 오른발 추가)
- 4교시
  - materials.urdf 작성 (색상 추가)
  - visual.urdf 작성 (얼굴, 그리퍼 추가)
- 5교시
  - flexible.urdf 작성 (type continuous, revolute, prismatic 추가)
  - mesh 파일 추가
