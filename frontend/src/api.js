const API_BASE_URL = "http://127.0.0.1:8000"; // Update this if your backend is hosted elsewhere

export const fetchPrediction = async (features) => {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ features }),
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data;
    } catch (error) {
        console.error("Error fetching prediction:", error);
        return null;
    }
};
