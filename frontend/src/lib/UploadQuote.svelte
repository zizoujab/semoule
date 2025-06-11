<script lang="ts">
    let file: File | null = null; // Type for the file
    let uploadMessage: string = "";
    let text:string = "";
    let success = false;
    let loading = false;
    let imageUrl;
    let imageLoaded = false;
    // Define the type for the API response
    type ApiResponse = {
        message: string;
        file?: string;
        error?: string;
    };

    // API URL (replace this with your actual API URL)
    const apiUrl = 'http://localhost:8090/api/invoice/ml';

    // Handle form submission
    async function uploadFile() {
        loading = true;
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

            // const data: ApiResponse = await response.json();
            // text = data?.text;
            if (response.ok) {
                uploadMessage = `File uploaded successfully !`;
                const blob = await response.blob();
                imageUrl = URL.createObjectURL(blob);
                success = true;

            } else {
                uploadMessage = `Error: ${data.error || 'Failed to upload the file'}`;
            }
        } catch (error) {
            uploadMessage = `Error: ${(error as Error).message}`;
        } finally {
            loading = false;
        }
    }

    function handleImageLoad() {
        imageLoaded = true;
    }
    import Zoom from 'svelte-zoom'
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
</div>
<div class="upload-container">
    <Zoom src="{imageUrl}" alt="Zoomable image" />
</div>
