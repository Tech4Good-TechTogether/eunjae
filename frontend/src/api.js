export const sendFrameToServer = async (imageSrc) => {
    try {
      const response = await fetch(`${process.env.REACT_APP_API_URL}/api/analyze_frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageSrc }),
      });
  
      const data = await response.json();
      console.log('Server response:', data);
    } catch (error) {
      console.error('Error sending frame to server:', error);
    }
  };
  