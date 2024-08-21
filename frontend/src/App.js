import React, { useRef, useEffect, useState } from 'react';
import Webcam from 'react-webcam';
import { sendFrameToServer, fetchGuideLandmarks } from './api';
import './App.css';

function App() {
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);
  const [guideLandmarks, setGuideLandmarks] = useState([]);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(0);

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
  }, []); // 빈 의존성 배열을 전달해 컴포넌트가 처음 렌더링될 때 한 번만 실행되도록 함

  // 웹캠에서 프레임을 캡처하여 서버로 전송하는 함수
  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      sendFrameToServer(imageSrc);
    } else {
      console.error("Failed to capture image.");
    }
  };

  // 웹캠에 랜드마크를 그리는 함수
  const drawLandmarks = (ctx, landmarks) => {
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.strokeStyle = 'red';
    ctx.lineWidth = 2;

    landmarks.forEach(landmark => {
      landmark.forEach(([x, y]) => {
        ctx.beginPath();
        ctx.arc(x * ctx.canvas.width, y * ctx.canvas.height, 5, 0, 2 * Math.PI);
        ctx.stroke();
      });
    });
  };

  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();

      if (guideLandmarks.length > 0) {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        // 현재 프레임의 랜드마크 그리기
        drawLandmarks(ctx, guideLandmarks[currentFrameIndex]);

        // 다음 프레임으로 이동 (순환)
        setCurrentFrameIndex(prevIndex => (prevIndex + 1) % guideLandmarks.length);
      }
    }, 100); // 100ms마다 프레임을 캡처하고, 가이드 랜드마크를 웹캠에 그리기

    return () => clearInterval(interval);
  }, [guideLandmarks, currentFrameIndex]);

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
