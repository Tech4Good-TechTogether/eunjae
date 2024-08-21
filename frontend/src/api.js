export async function fetchGuideLandmarks() {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  const response = await fetch(`${apiUrl}/api/get_guide_landmarks`);
  if (!response.ok) {
    throw new Error('Failed to fetch guide landmarks');
  }
  
  const data = await response.json();
  console.log("guide 영상 랜드마크 확인", data);
  
  return data;
}

export const sendFramesToServer = async (imageSrcList) => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  try {
    const response = await fetch(`${apiUrl}/api/analyze_frame`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ images: imageSrcList }),  // 여러 이미지를 서버로 전송
    });

    const data = await response.json();
    console.log("비교결과 출력 렛츠고",data["comparison"]);
    // if (data.some(result => result.status === "Hand detected")) {
    //   // 추가 로직을 여기에 작성할 수 있습니다.
    //   console.log("손동작이 감지되었습니다.")
    // }
    console.log('Server response:', data);
  } catch (error) {
    console.error('Error sending frames to server:', error);
  }
};

