<script>
  import { navigate } from 'svelte-navigator';
  
  let file;
  let fileInput;
  let dragging = false;
  let uploading = false;
  let error = null;
  let useGemini = true; // Default to using Gemini analysis
  
  // Server API URL
  const API_URL = 'http://localhost:5001/api';
  
  // Handle file drop
  function handleDrop(event) {
    dragging = false;
    event.preventDefault();
    
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      file = event.dataTransfer.files[0];
    }
  }
  
  // Prevent default drag behavior
  function handleDragOver(event) {
    dragging = true;
    event.preventDefault();
  }
  
  // Reset drag state when leaving the drop area
  function handleDragLeave() {
    dragging = false;
  }
  
  // Handle file input change
  function handleFileChange(event) {
    if (event.target.files && event.target.files.length > 0) {
      file = event.target.files[0];
    }
  }
  
  // Handle file upload
  async function uploadImage() {
    if (!file) {
      error = 'Please select a file';
      return;
    }
    
    // Check file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!validTypes.includes(file.type)) {
      error = 'Invalid file type. Please upload a JPEG or PNG image.';
      return;
    }
    
    // Reset error
    error = null;
    uploading = true;
    
    try {
      // Create FormData object
      const formData = new FormData();
      formData.append('file', file);
      formData.append('use_gemini', useGemini.toString());
      
      console.log(API_URL);
      // Send request to server
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process image');
      }
      
      // Get result data
      const resultData = await response.json();
      
      // Navigate to results page with data
      navigate('/results', { state: { result: resultData } });
    } catch (err) {
      console.error('Upload error:', err);
      error = err.message || 'Failed to upload image';
    } finally {
      uploading = false;
    }
  }
</script>

<div class="card">
  <h2 class="text-center">Mammogram Image Analysis</h2>
  
  <div class="mb-4">
    <p>Welcome to Neuralrad Mammo AI. This advanced tool uses multiple AI models to detect, classify, and interpret masses in mammogram images.</p>
    
    <h3>How it works:</h3>
    <ol>
      <li>Upload a mammogram image (JPEG or PNG format)</li>
      <li>The AI models will process the image and detect suspicious areas</li>
      <li>View the results with annotations and detailed interpretations</li>
    </ol>
  </div>
  
  <!-- File upload area -->
  <div 
    class="p-4 mb-4" 
    style="border: 2px dashed {dragging ? '#4a76a8' : '#ccc'}; border-radius: 8px; background-color: {dragging ? '#f0f6ff' : 'transparent'}; transition: all 0.3s"
    on:drop={handleDrop}
    on:dragover={handleDragOver}
    on:dragleave={handleDragLeave}
  >
    <div class="text-center">
      <h3>Drop Mammogram Image Here</h3>
      <p>or</p>
      <button class="btn" on:click={() => fileInput.click()}>Select File</button>
      <input 
        type="file" 
        accept=".jpg,.jpeg,.png" 
        style="display: none" 
        bind:this={fileInput} 
        on:change={handleFileChange}
      />
      
      {#if file}
        <p class="mt-4">Selected file: {file.name}</p>
      {/if}
    </div>
  </div>
  
  <!-- Error message -->
  {#if error}
    <div class="p-4 mb-4" style="background-color: #ffebee; color: #b71c1c; border-radius: 4px;">
      <p>{error}</p>
    </div>
  {/if}
  
  <!-- Upload button -->
  <div class="text-center">
    <!-- Gemini option checkbox -->
    <div class="p-3 mb-3" style="background-color: #f3e5f5; border-radius: 4px;">
      <label style="display: flex; align-items: center; justify-content: center; gap: 8px; cursor: pointer;">
        <input type="checkbox" bind:checked={useGemini} style="width: 16px; height: 16px;">
        <span>Use Neuralrad AI for advanced mammogram interpretation</span>
      </label>
      {#if useGemini}
        <div class="mt-2" style="font-size: 0.9em; color: #6a1b9a;">
          Neuralrad AI will provide detailed radiologist-like analysis and check for additional suspicious areas
        </div>
      {/if}
    </div>
    
    <button 
      class="btn btn-primary" 
      on:click={uploadImage} 
      disabled={!file || uploading}
    >
      {#if uploading}
        Processing...
      {:else}
        Analyze Mammogram
      {/if}
    </button>
  </div>
</div>
