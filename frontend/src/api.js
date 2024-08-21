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

export const sendFrameToServer = async (imageSrc) => {
  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';
  try {
    const response = await fetch(`${apiUrl}/api/analyze_frame`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: imageSrc }),
    });

    const data = await response.json();
    if (data.status === "Hand detected") {
      // 추가 로직을 여기에 작성할 수 있습니다.
      console.log("손동작이 감지되었습니다.")
    }
    console.log('Server response:', data);
  } catch (error) {
    console.error('Error sending frame to server:', error);
  }
};
