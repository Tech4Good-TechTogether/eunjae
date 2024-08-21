// import logo from './logo.svg';
// import './App.css';

// function App() {
//   return (
//     <div className="App">
//       <header className="App-header">
//         <img src={logo} className="App-logo" alt="logo" />
//         <p>
//           Edit <code>src/App.js</code> and save to reload.
//         </p>
//         <a
//           className="App-link"
//           href="https://reactjs.org"
//           target="_blank"
//           rel="noopener noreferrer"
//         >
//           Learn React
//         </a>
//       </header>
//     </div>
//   );
// }

// export default App;

import React, { useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { sendFrameToServer } from './api';
import './App.css';

function App() {
  const webcamRef = useRef(null);

  const captureFrame = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    // console.log("Captured Image URL:", imageSrc);
    if (imageSrc) {
      sendFrameToServer(imageSrc);
    } else {
      console.error("Failed to capture image.");
    }
  };


  useEffect(() => {
    const interval = setInterval(() => {
      captureFrame();
    }, 100); // 100ms마다 프레임을 캡처하여 서버로 전송

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="App">
      <h1>Real-Time Rehabilitation Analysis</h1>
      <Webcam 
        ref={webcamRef} 
        screenshotFormat="image/jpeg" 
        videoConstraints={{ width: 640, height: 480, facingMode: "user" }}
      />
    </div>
  );
}

export default App;
