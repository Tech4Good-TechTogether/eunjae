import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { sendFramesToServer, fetchGuideLandmarks } from './api';
import './App.css';

// Mediapipe 손가락 랜드마크 인덱스 구조 정의
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // 엄지
  [0, 5], [5, 6], [6, 7], [7, 8], // 검지
  [5, 9], [9, 10], [10, 11], [11, 12], // 중지
  [9, 13], [13, 14], [14, 15], [15, 16], // 약지
  [13, 17], [17, 18], [18, 19], [19, 20], // 새끼
  [0, 17] // 손바닥 연결
];

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [guideLandmarks, setGuideLandmarks] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);
  const [capturedFrames, setCapturedFrames] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);

  // 가이드 영상의 랜드마크 데이터를 서버로부터 한 번만 가져오는 함수
  useEffect(() => {
    const loadGuideLandmarks = async () => {
      try {
        const response = await fetchGuideLandmarks();
        const { guide_landmarks } = response;
        setGuideLandmarks(guide_landmarks);
      } catch (error) {
        console.error("Failed to load guide landmarks:", error);
      }
    };

    loadGuideLandmarks();
  }, []);

  // 3초 동안 프레임을 캡처하는 함수
  const captureFramesForDuration = () => {
    setIsCapturing(true);
    const captureInterval = 100; // 100ms마다 프레임 캡처
    const duration = 3000; // 3초

    let frames = [];
    const intervalId = setInterval(() => {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        frames.push(imageSrc);
      }
    }, captureInterval);

    setTimeout(() => {
      clearInterval(intervalId);
      setCapturedFrames(frames);
      setIsCapturing(false);
      sendFramesToServer(frames);
    }, duration);
  };


  // 웹캠에 랜드마크를 그리는 함수
  const drawLandmarks = (ctx, landmarks) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    // 선 스타일
    ctx.strokeStyle = '#00FF00'; // 연한 초록색 선
    ctx.lineWidth = 3;

    // 점 스타일
    ctx.fillStyle = '#FF4500'; // 주황색 점
    const pointRadius = 6;

    landmarks.forEach(landmark => {
      // 점 그리기
      landmark.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * ctx.canvas.width, y * ctx.canvas.height, pointRadius, 0, 2 * Math.PI);
        ctx.fill();
      });

      // 선 그리기
      HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
        const startX = landmark[startIdx][0] * ctx.canvas.width;
        const startY = landmark[startIdx][1] * ctx.canvas.height;
        const endX = landmark[endIdx][0] * ctx.canvas.width;
        const endY = landmark[endIdx][1] * ctx.canvas.height;

        ctx.beginPath();
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
        ctx.stroke();
      });
    });
  };

  useEffect(() => {
    const interval = setInterval(() => {
      if (!isCapturing) {
        captureFramesForDuration();
      }

      if (guideLandmarks.length > 0) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // 현재 프레임의 랜드마크 그리기
        drawLandmarks(ctx, guideLandmarks[currentFrameIndex]);

        // 다음 프레임으로 이동 (순환)
        setCurrentFrameIndex(prevIndex => (prevIndex + 1) % guideLandmarks.length);
      }
    }, 100);

    return () => clearInterval(interval);
  }, [guideLandmarks, currentFrameIndex, isCapturing]);

  return (
    <div className="App">
      <h1>Real-Time Rehabilitation Analysis</h1>
      <div style={{ position: 'relative' }}>
        <Webcam
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
        />
        <canvas
          ref={canvasRef}
          id="landmarksCanvas"
          width="640"
          height="480"
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            zIndex: 2,
            pointerEvents: 'none'
          }}
        />
      </div>
    </div>
  );
}

export default App;
