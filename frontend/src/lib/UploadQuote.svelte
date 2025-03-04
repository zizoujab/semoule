<script lang="ts">
    let file: File | null = null; // Type for the file
    let uploadMessage: string = "";
    let text:string = "";

    // Define the type for the API response
    type ApiResponse = {
        message: string;
        file?: string;
        error?: string;
    };

    // API URL (replace this with your actual API URL)
    const apiUrl = 'http://localhost:8081/api/invoice';

    // Handle form submission
    async function uploadFile() {
        if (!file) {
            uploadMessage = "Please select a file before uploading.";
            return;
        }

        // Create FormData object to send the file
        const formData = new FormData();
        formData.append('image', file);

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData
            });

            const data: ApiResponse = await response.json();
            text = data?.text;
            if (response.ok) {
                uploadMessage = `File uploaded successfully: ${data.file}`;
            } else {
                uploadMessage = `Error: ${data.error || 'Failed to upload the file'}`;
            }
        } catch (error) {
            uploadMessage = `Error: ${(error as Error).message}`;
        }
    }
</script>

<style>
    .upload-container {
        margin-top: 20px;
    }

    .upload-message {
        margin-top: 10px;
        color: green;
    }

    .error-message {
        color: red;
    }
</style>

<div class="upload-container">
    <h2>Upload a Quote Image</h2>

    <!-- Input field for selecting the file -->
    <input type="file" on:change="{(e) => file = e.target.files ? e.target.files[0] : null}" accept="image/*" />

    <!-- Upload button -->
    <button on:click="{uploadFile}">Upload</button>

    <!-- Display the message (success or error) -->
    {#if uploadMessage}
        <p class:upload-message="{!uploadMessage.startsWith('Error')}">{uploadMessage}</p>
        <p class="api_response">{text}</p>
    {/if}
</div>