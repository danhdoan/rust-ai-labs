document.getElementById("generateBtn").addEventListener("click", async function() {
    const prompt = document.getElementById("prompt").value;
    const negativePrompt = document.getElementById("negativePrompt").value.trim();
    const loading = document.getElementById("loading");
    const image = document.getElementById("image");

    if (!prompt) {
        alert("Please enter a prompt!");
        return;
    }

    loading.classList.remove("hidden");
    image.classList.add("hidden");
    image.classList.remove("loaded");

    try {
        const response = await fetch("http://10.0.30.32:8000/generate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              prompt: prompt,
              neg_prompt: negativePrompt
            })
        });

        if (!response.ok) {
            throw new Error("Failed to generate image");
        }

        const data = await response.json();
        image.src = `data:image/png;base64,${data.image}`;
        image.classList.remove("hidden");

        // Wait for image to load, then apply animation
        image.onload = () => {
            image.classList.add("loaded");
        };
    } catch (error) {
        console.error("Error:", error);
        alert("Error generating image!");
    } finally {
        loading.classList.add("hidden");
    }
});
