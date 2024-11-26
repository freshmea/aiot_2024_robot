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
  - rviz config 파일 작성 ( urdf.rviz )
- 3교시
  - multipleshapes.urdf 작성 (link 여러개 추가)
  - origins.urdf 작성 origin 설정( 왼발 오른발 추가)
- 4교시
  - materials.urdf 작성 (색상 추가)
  - visual.urdf 작성 (얼굴, 그리퍼 추가)
- 5교시
  - flexible.urdf 작성 (type continuous, revolute, prismatic 추가)
  - mesh 파일 추가
  - [gazebo plugin link](https://classic.gazebosim.org/tutorials?tut=ros_gzplugins#AddingaModelPlugin)
- 6교시
  - inertial 설명 ( collision, mass, inertia, origin 추가)
  - physics.urdf 작성
  - gazebo plugin 설명
  - gazebo plugin 작성 예시(turtlebot3_simulation 분석)
- 7교시
  - xacro 설명, 기본 문법
  - xacro 예제 (xacroed.urdf.xacro 작성)

---

## 2024_11_07

---

- 1교시
  - 복습
  - turtlesim_turtle_tf 실습(simple_ros_cpp)
  - dynamic_tf2 작성
- 2교시
  - turtlesim, dynamic_tf, dynamic_tf2, follow_turtle_tf 노드 실행
- 3교시
  - move_turtle 에 follow_wall 노드 수정(python)
  - follow_point tf 발행 후, follow_point 추적 하여 움직이는 노드 작성
  - gazebo 로 작동 확인
- 4교시
  - lidar 센서에서 0.0 인 경우 처리
  - 터틀봇3 기체 적용
- 5교시
  - /map 토픽 OccupiedGridMap 메시지 분석
  - publish_map 노드 작성
- 6교시
  - /map 토픽의 msg.data 분석
  - data -1: unknown, 0: free, 100: occupied
- 7교시
  - /map 토픽 lidar 데이터로 맵 업데이트 (publish_map_with_lidar 노드 작성)
  - cartographer 실습

---

## 2024_11_08

---

- 1교시
  - cartographer 실습 및 패키지 설명
  - tunning 작업 ( imu enable, etc)
- 2교시
  - 맵 파일로 저장
  - navigation2 실습
- 3교시
  - navigation2 과 cartographer 연동 실습
  - turtlebot3 기체로 navigation2 실습
- 4교시
  - cartographer 없이 map_server 로 저장된 맵 파일로 navigation2 실습
  - **에러 해결** :
    - odom 과 map 사이의 static_tf 발행:
    - map_server lifecycle activate:
  - navigation 실습

```bash
ros2 run tf2_ros static_transform_publisher --x -2.0 --y -0.5 --z 0 --yaw 0 --pitch 0 --roll 0 --frame-id odom --child-frame-id map --ros-args -p use_sim_time:=true
ros2 lifecycle set /map_server activate
```

- 5교시
  - navigation2 설명
- 6교시
  - navigation2 action 코드 작성(follow_waypoints)
  - patrol 노드 작성(follow_waypoints_loop)

---

## 2024_11_11

---

- 1교시
  - 카메라 설정
- 2교시
  - 터틀봇3 카메라 설정
- 3교시
  - 터틀봇3 런치 파일 작성( robot.launch.py + raspicam.launch.py)
- 4교시
  - canny 노드 작성( CompressedImage 토픽 구독)
- 5교시
  - r2d2 urdf 에 카메라 노드 추가
- 6교시
  - turtlebot3의 gazebo sdf 파일 작성 ( 기존의 turtlebot_bruger/model.sdf 변경)
- 7교시
  - camera simulation

---

## 2024_11_12

---

- 1교시
  - 복습
  - aruco 노드 패키지 다운로드 및 실행
  - aruco marker 생성
- 2교시
  - sudo apt install ros-humble-tf-transformations 설치
  - aruco 코드 수정
  - numpy 버전 에러 2.대에서 1.26으로 다운 그레이드
- 3교시
  - generate_aruco_marker.py 실행 후 파일 프린트
  - image_transport republish 노드로 image 토픽 변환
  - aruco 런치 파일 작성
- 4교시
  - ar 마커 카메라에 확인 후 aruco_markers 토픽 확인.
  - 시뮬레이션에 ar marker 추가
- 5교시
  - 시뮬레이션에 aruco 마커 추가 - autorace 표지판에 추가
  - aruco 마커에 tf 추가 후 aruco 마커 따라가기 (gazebo 시뮬레이션)
  - TODO: aruco 마커의 tf frame 이 optical 인거 같음.
  - TODO: turtlebot3_burger.urdf 파일은 변경하지 않아서 tf tree 가 완성 안됨.
- 6교시
  - led_test.py 작성
  - led_server 서비스 노드 작성
- 7교시
  - servo_test.py 작성

---

## 2024_11_13

---

- 1교시
  - 복습
  - aruco 마커 따라가기 코드 수정
  - turtlebot3_gazebo 패키지의 turtlebot3_burger.urdf 수정 ( camera_link_optical 추가)
- 2교시
  - servo_server 노드 작성
  - 노트북에서 rqt 로 servo_server 제어
- 3교시
  - servo_sub 노드 작성
  - 노트북에서 rqt 로 servo_sub 제어
- 4교시
  - servo cpp 노드 작성
- 5교시
  - servo cpp 노드 작성 (wiringPi 사용)
  - wiringpi.h가 안 찾아 지는 문제 -> sudo apt install libwiringpi-dev 설치 해결
- 6교시
  - servo_test.cpp 작성 -> 모터 동작 확인
  - ros2 run simple_ros_cpp servo_cpp 실행 -> 권한 설정 문제
  - sudo chown root:root install/move_turtle_cpp/lib/move_turtle_cpp/servo_sub
  - sudo chmod +s install/move_turtle_cpp/lib/move_turtle_cpp/servo_sub
    - librcl_yaml_param_parser.so: cannot open shared object file 에러 발생 ??
    - sudo 설정시 PATH 문제로 발생 한거 같음.
- 7교시
  - 권한 설정 문제가 없는 pigpio 라이브러리 사용
    - [링크](https://github.com/botamochi6277/ros2_pigpio?tab=readme-ov-file) 참조
    - 소스 받아서 빌드 후 사용 -> so 파일 못 찾음. -> path 설정 문제 해결하고 넘어가야 할듯.
  - arduino linux 설정
  - arduino 확장 설치

---

## 2024_11_14

---

- 1교시
  - wiringpi 에러 해결
    - sudo -E bash -c '...' 로 권한 문제 해결
  - pigpio 에러 해결
    - ldconfig 로 so 파일 path 해결
- 2교시
  - led.ino 파일 작성
- 3교시
  - arduino_led.py 노드 작성 (아두이노 시리얼 통신 노드)
- 4교시
  - 터틀봇3에 아두이노 연결 토픽 확인
- 5교시
  - 아두이노 switch.ino 파일 작성 falling edge rising edge
- 6교시
  - 스위치 작성 arduino_switch 노드 작성
- 7교시
  - servo.ino 작성 ( 서보 모터 동작 확인)

---

## 2024_11_15

---

- 1교시
  - 복습
  - arduino 외부 라이브러리 활용
  - switch class 작성 (스위치 여러개 쓰기)
- 2교시
  - ros2 노드 arduino_servo 작성
- 3교시
  - urdf 를 이용해서 터틀봇에 서보 모터 추가(깃발)
- 4교시
  - arduino_servo 에서 tf 를 발행하는 코드 추가( 서보의 동작과 동시에 tf 발행)
- 5교시
  - namespace 설명
  - turtlebot3 로봇의 bringup robot.launch.py 에서 수정 해야 할 부분 설명
  - move_turtle 의 launch 폴더에 myrobot1.launch.py 등을 작성
  - burger.yaml 을 namespace 전용으로 작성
- 6교시
  - /scan topic namespace 변경 -> 따로 launch 파일 작성
  - 두 개의 turtlebot3 namepace 적용 실습
  - TODO: tf 의 이름 바꾸는 실습 진행 안 함

---

## 2024_11_18

---

- 1교시
  - 복습
  - tf 명 변경 ( myrobot1)
- 2교시
  - xacro 로 urdf 작성 실습 namespace property 추가
  - openmanipulator 기기 조립 (일부 조립 되어 있고 해체 되어 있음)
- 3교시
  - openmanipulator 기기 조립
- 4교시
  - open manipulator 실습
- 5교시
  - open manipulator teleop 실습
- 6교시
  - open_manipulator teleop 움직임 노드 작성
- 7교시
  - open_manipulator patrol 코드 작성

---

## 2024_11_19

---

- 1교시
  - moveit2 설치

```bash
sudo apt install python3-rosdep
sudo rosdep init
rosdep update
sudo apt update
sudo apt dist-upgrade
sudo apt install python3-colcon-common-extensions
sudo apt install python3-colcon-mixin
colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml
colcon mixin update default
sudo apt install python3-vcstool
cd ~/ws_moveit/src
git clone -b humble https://github.com/moveit/moveit2_tutorials
vcs import --recursive < moveit2_tutorials/moveit2_tutorials.repos
sudo apt remove ros-humble-moveit*
sudo apt update && rosdep install -r --from-paths . --ignore-src --rosdistro humble -y
cd ~/ws_moveit
colcon build --mixin release
```

- 2교시
  - moveit2 설치
- 3교시
  - moveit2 설치
  - 라즈베리파이 피코 납땜
- 4교시
  - moveit2 설치
  - 라즈베리파이 피코 납땜
- 5교시
  - moveit2 rviz2 실습
- 6교시
  - cpp 코드로  moveit2 실습 (move_turtle / hello_moveit)
- 7교시
  - mtc example 설명
  - 터틀봇3 manipulation 설명

---

## 2024_11_20

---

- 1교시
  - 복습
  - micro-ros 설치
- 2교시
  - 라즈베리파이 피코 보드 설명
- 3교시
  - 라즈베리파이 피코 보드 - 마이크로 로스 연결
  - led 예제 (마이크로 로스 publisher 실습)
- 4교시
  - 마이크로 로스 Subscription 실습
- 5교시
  - 마이크로 로스 서보 모터 제어 실습 (외부 라이브러리 사용 예제)
  - [참조](https://github.com/irishpatrick/pico-servo.git)
  - libpico_servo.a, libfixmath.a 를 사용하기 위해 /usr/bin 에 복사
- 6교시
  - 프로젝터 교체 (30분 휴식)
- 7교시
  - dc 모터 제어 ( 모터 드라이버 사용)
  - dc 모터 제어 노드 작성 ( rclc 코드 )

---

## 2024_11_21

---

- 1교시
  - 복습
  - dc 모터 pwm 제어 노드 작성 (micro-ros pico)
- 2교시
  - service led 제어 노드 작성 (micro-ros pico)
- 3교시
  - ic2 통신 설명
  - lcd arduino 코드 작성 (i2c 통신)
- 4교시
  - lcd 제어 노드 작성 (ros2 desktop)
- 5교시
  - 터틀봇3에 아두이노-lcd 추가
  - lcd 제어 노드 작성 (터틀봇3 move_turtle lcd_server)
- 6교시
  - 터틀봇3에 gpio - lcd 연결(logicl level converter 사용)
- 7교시
  - 피코 보드에 lcd 연결 (i2c 통신), micro-ros 로 service 생성 및 lcd 제어
  - [참고 lcd c library](https://github.com/extremq/lcd1602_i2c/blob/main/lcd1602_i2c.c)
  - [참고]

---

## 2024_11_22

---

- 1교시
  - dds 설정 ( discovery server 생성)
- 2교시
  - discovery backup server 설정
  - super_client_configuration_file.xml 설정
- 3교시
  - topic syn/async 설정 (dds)
  - qos 설정 (dds)
- 4교시
  - 3D 프린터 설명 (bambu lab carbon x1 printer)
- 5교시
  - ros2 학습 체크 리스트 작성
  - 3D 프린터 이름표 실습(tinker cad)
- 6교시
  - 프로젝트 안내
  - 주변 정리

---

## 2024_11_25

---

- 프로젝트 시작
- 안내
  - 팀 github repository 생성 및 운용
    - README.md 작성(팀장이 오전 오후 개발 로그 작성)
  - 오전 9시에는 모두 모여서 출석 확인
  - 11/28 오후 4시 계획발표 (ppt)

<!-- TODO : VsCode 확장 세팅 -->
<!-- TODO : SROS 활성화 및 작동 확인 -->
<!-- TODO : 메뉴퓰레이터 그리퍼 문제 -->
<!-- TODO : 파라미터 Declare 및 get, set, event 에 의한 변경 함수 고안 -->
<!-- TODO : 인터페이스 교재 및 교안 작성 -->
<!-- TODO : DDS discovery server 를 활용한 클라우드 ROS2 구현 실습 -->
<!-- TODO : micro-ros 에서 SROS 나 보안 강화를 위한 방법이 있는지 검색 -->
- enclave 설정 완료  -> 환경 변수에 오타가 있었음 keystone -> keystore
- 메뉴퓰레이터 그리퍼 안 움직이던 것 -> 그리퍼 코드가 안 들어 있음 -> 코드 수정 후 올림

- cloud test
  - fastdds discovery -i 1 -l 127.0.0.1 -p 11811
  - fastdds discovery -i 1 -l <AWS_PUBLIC_IP> -p 11811
  - fastdds discovery -i 1 -l 0.0.0.0 -p 11811

  - ros2 deamon stop
  - ros2 deamon start
  - 파일 생성

```xml
<profiles>
    <participant profile_name="participant_profile" is_default_profile="true">
        <rtps>
            <builtin>
                <discovery_config>
                    <discovery_protocol>SERVER</discovery_protocol>
                    <static_edp>true</static_edp>
                    <ignoreParticipantFlags>0</ignoreParticipantFlags>
                    <metatraffic_unicast_locator_list>
                        <locator>
                            <kind>1</kind>
                            <address><AWS_PUBLIC_IP></address>
                            <port>11811</port>
                        </locator>
                    </metatraffic_unicast_locator_list>
                </discovery_config>
            </builtin>
        </rtps>
    </participant>
</profiles>
```

- export FASTRTPS_DEFAULT_PROFILES_FILE=super_client_configuration_file.xml
- export ROS_DISCOVERY_SERVER=ec2-54-180-113-6.ap-northeast-2.compute.amazonaws.com:11811
