const API_URL = "http://127.0.0.1:8000";

export const predictMarket = async (features) => {
    try {
        const response = await fetch(`${API_URL}/predict`, {  // Corrected URL usage
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
        console.error("Error calling API:", error);
        return null;
    }
};