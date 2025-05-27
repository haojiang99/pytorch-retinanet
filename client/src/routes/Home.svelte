<script>
  import { navigate } from 'svelte-navigator';
  
  // State
  let file;
  let fileInput;
  let dragging = false;
  let uploading = false;
  let error = null;
  let useGemini = true; // Default to using Gemini analysis
  let currentTab = 'upload'; // 'upload' or 'workflow'
  let filePreview = null;
  
  // Server API URL
  const API_URL = 'http://mammo.neuralrad.com:5300/api';
  
  // Handle file drop
  function handleDrop(event) {
    dragging = false;
    event.preventDefault();
    
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      file = event.dataTransfer.files[0];
      createFilePreview(file);
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
      createFilePreview(file);
    }
  }
  
  // Create file preview
  function createFilePreview(file) {
    if (!file || !file.type.startsWith('image/')) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      filePreview = e.target.result;
    };
    reader.readAsDataURL(file);
  }
  
  // Clear selected file
  function clearFile() {
    file = null;
    filePreview = null;
    if (fileInput) fileInput.value = '';
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
  
  // Workflow steps data
  const workflowSteps = [
    {
      title: "Upload Mammogram",
      description: "Upload a mammogram image (JPEG or PNG format) for research analysis.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`
    },
    {
      title: "Deep Learning Analysis",
      description: "The RetinaNet model processes the image to identify and classify potential regions of interest for research purposes.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line></svg>`
    },
    {
      title: "Vision LLM Interpretation",
      description: "Advanced vision language model provides detailed analysis and interpretation of the detected regions for research investigation.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>`
    },
    {
      title: "Research Results",
      description: "Review the analysis results with interactive tools and detailed AI-generated insights for research purposes only.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="8" y1="11" x2="14" y2="11"></line><line x1="11" y1="8" x2="11" y2="14"></line></svg>`
    }
  ];
  
  function switchTab(tab) {
    currentTab = tab;
  }
</script>

<style>
  .hero {
    background: linear-gradient(135deg, rgba(233, 30, 99, 0.9) 0%, rgba(156, 39, 176, 0.9) 100%);
    color: white;
    padding: 4rem 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    text-align: center;
  }
  
  .hero h1 {
    font-size: 3rem;
    margin-bottom: 1rem;
    font-weight: 700;
  }
  
  .hero p {
    font-size: 1.2rem;
    max-width: 800px;
    margin: 0 auto 2rem auto;
    opacity: 0.9;
  }
  
  .tab-buttons {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
  }
  
  .tab-button {
    padding: 0.75rem 1.5rem;
    border: none;
    background-color: rgba(255, 255, 255, 0.2);
    color: white;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 500;
    transition: all 0.3s ease;
  }
  
  .tab-button.active {
    background-color: white;
    color: #e91e63;
  }
  
  .tab-button:hover:not(.active) {
    background-color: rgba(255, 255, 255, 0.3);
  }
  
  .workflow-step {
    display: flex;
    align-items: center;
    gap: 1rem;
    background-color: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    margin-bottom: 1.5rem;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
  }
  
  .workflow-step:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
  }
  
  .step-number {
    position: absolute;
    top: -10px;
    left: -10px;
    width: 32px;
    height: 32px;
    background-color: #e91e63;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
  }
  
  .step-icon {
    background-color: #fce4ec;
    color: #e91e63;
    padding: 1rem;
    border-radius: 12px;
    flex-shrink: 0;
  }
  
  .step-content h3 {
    margin: 0 0 0.5rem 0;
    color: #333;
  }
  
  .step-content p {
    margin: 0;
    color: #666;
  }
  
  .upload-area {
    border: 2px dashed #ce93d8;
    border-radius: 8px;
    padding: 3rem 2rem;
    text-align: center;
    transition: all 0.3s ease;
    background-color: #fce4ec;
  }
  
  .upload-area.dragging {
    border-color: #e91e63;
    background-color: #f8bbd0;
  }
  
  .upload-area h3 {
    margin-top: 0;
    color: #c2185b;
  }
  
  .upload-icon {
    font-size: 3rem;
    color: #e91e63;
    margin-bottom: 1rem;
  }
  
  .example-card {
    background-color: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
  }
  
  .example-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
  }
  
  .example-card img {
    width: 100%;
    height: 300px;
    object-fit: cover;
    border-bottom: 2px solid #f8bbd0;
  }
  
  .example-caption {
    padding: 1rem;
    text-align: center;
  }
  
  .example-caption h3 {
    margin-top: 0;
    color: #c2185b;
  }
  
  .example-caption p {
    color: #666;
    margin-bottom: 0;
  }
  
  .option-checkbox {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    cursor: pointer;
    padding: 1rem;
    background-color: #fce4ec;
    border-radius: 4px;
    margin-bottom: 1rem;
    transition: background-color 0.3s ease;
  }
  
  .option-checkbox:hover {
    background-color: #f8bbd0;
  }
  
  .option-checkbox input {
    width: 18px;
    height: 18px;
    accent-color: #e91e63;
  }
  
  .option-description {
    margin-top: 0.5rem;
    font-size: 0.9em;
    color: #6a1b9a;
    padding: 0 1rem;
  }
  
  .error-message {
    background-color: #ffebee;
    color: #b71c1c;
    border-radius: 4px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  
  .upload-button {
    margin-top: 1rem;
    width: 100%;
    max-width: 300px;
  }
  
  .examples-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(290px, 1fr));
    gap: 1.5rem;
    margin-top: 3rem;
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .select-file-btn {
    background: linear-gradient(135deg, #e91e63 0%, #c2185b 100%);
    color: white;
    border: none;
    padding: 12px 32px;
    border-radius: 30px;
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    box-shadow: 0 4px 10px rgba(233, 30, 99, 0.3);
    position: relative;
    overflow: hidden;
  }
  
  .select-file-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(233, 30, 99, 0.4);
  }
  
  .select-file-btn:active {
    transform: translateY(0);
    box-shadow: 0 2px 5px rgba(233, 30, 99, 0.3);
  }
  
  .select-file-btn::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: linear-gradient(rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0));
    border-radius: 30px;
  }
  
  .logo-container {
    margin-bottom: 1.5rem;
  }
  
  .logo {
    max-width: 200px;
    height: auto;
    border: none;
    box-shadow: none;
  }
  
  .file-preview-container {
    margin-top: 1.5rem;
    position: relative;
    max-width: 400px;
    margin-left: auto;
    margin-right: auto;
  }
  
  .file-preview {
    width: 100%;
    height: auto;
    max-height: 300px;
    object-fit: contain;
    border-radius: 8px;
    border: 3px solid #f8bbd0;
    background-color: #fff;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  
  .file-preview-caption {
    margin-top: 0.5rem;
    font-size: 0.9rem;
    color: #c2185b;
    text-align: center;
  }
  
  .clear-file-btn {
    position: absolute;
    top: -10px;
    right: -10px;
    width: 30px;
    height: 30px;
    border-radius: 50%;
    background-color: #e91e63;
    color: white;
    border: 2px solid white;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
  }
  
  .clear-file-btn:hover {
    background-color: #c2185b;
    transform: scale(1.1);
  }
  
  .analyze-btn {
    background: linear-gradient(135deg, #e91e63 0%, #9c27b0 100%);
    color: white;
    font-size: 1.1rem;
    font-weight: 600;
    padding: 14px 40px;
    border: none;
    border-radius: 50px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(233, 30, 99, 0.4);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    letter-spacing: 0.5px;
    margin: 2rem auto;
    position: relative;
    overflow: hidden;
    width: 280px;
  }
  
  .analyze-btn:hover {
    transform: translateY(-3px);
    box-shadow: 0 7px 25px rgba(233, 30, 99, 0.5);
  }
  
  .analyze-btn:active {
    transform: translateY(-1px);
  }
  
  .analyze-btn::after {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: 0.5s;
  }
  
  .analyze-btn:hover::after {
    left: 100%;
  }
  
  .analyze-btn-container {
    display: flex;
    justify-content: center;
    margin-top: 2rem;
    width: 100%;
  }
  
  .disclaimer {
    background-color: #fff3e0;
    border: 2px solid #ff9800;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 2rem 0;
    color: #e65100;
    font-size: 0.9rem;
    line-height: 1.5;
  }
  
  .disclaimer h4 {
    margin-top: 0;
    margin-bottom: 1rem;
    color: #bf360c;
    font-weight: 600;
  }
  
  .disclaimer p {
    margin-bottom: 0.5rem;
  }
  
  .disclaimer strong {
    font-weight: 600;
    color: #bf360c;
  }

  @media (max-width: 768px) {
    .hero h1 {
      font-size: 2.2rem;
    }
    
    .hero p {
      font-size: 1rem;
    }
    
    .workflow-step {
      flex-direction: column;
      text-align: center;
    }
    
    .step-number {
      position: static;
      margin-bottom: 1rem;
    }
  }
</style>

<div class="hero">
  <div class="logo-container">
    <img src="/examples/NeuralRadLogo.png" alt="NeuralRad Logo" class="logo" />
  </div>
  <h1>Neuralrad Mammo AI</h1>
  <p>AI-powered mammogram analysis leveraging deep learning and vision LLM for research investigation</p>
  
  <div class="tab-buttons">
    <button 
      class="tab-button {currentTab === 'upload' ? 'active' : ''}" 
      on:click={() => switchTab('upload')}
    >
      Upload Mammogram
    </button>
    <button 
      class="tab-button {currentTab === 'workflow' ? 'active' : ''}" 
      on:click={() => switchTab('workflow')}
    >
      How It Works
    </button>
  </div>
</div>

<!-- FDA Disclaimer -->
<div class="disclaimer">
  <h4>⚠️ Important Research Disclaimer</h4>
  <p><strong>This application is NOT FDA 510(k) cleared.</strong></p>
  <p>This tool is designed for <strong>research investigation purposes only</strong> and is not intended for clinical diagnosis, treatment decisions, or patient care.</p>
  <p>All results generated by this AI system should be interpreted by qualified medical professionals and are not a substitute for professional medical judgment.</p>
  <p>Do not use this application for actual medical diagnosis or treatment decisions.</p>
</div>

{#if currentTab === 'upload'}
  <div class="card">
    <div 
      class="upload-area {dragging ? 'dragging' : ''}" 
      on:drop={handleDrop}
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
    >
      {#if file && filePreview}
        <!-- File preview -->
        <div class="file-preview-container">
          <button class="clear-file-btn" on:click={clearFile}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
          <img src={filePreview} alt="Selected mammogram" class="file-preview" />
          <div class="file-preview-caption">
            {file.name} ({Math.round(file.size / 1024)} KB)
          </div>
        </div>
      {:else}
        <div class="upload-icon">
          <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="17 8 12 3 7 8"></polyline>
            <line x1="12" y1="3" x2="12" y2="15"></line>
          </svg>
        </div>
        <h3>Drop Mammogram Image Here</h3>
        <p>or</p>
        <button class="select-file-btn" on:click={() => fileInput.click()}>
          <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
            <circle cx="8.5" cy="8.5" r="1.5"></circle>
            <polyline points="21 15 16 10 5 21"></polyline>
          </svg>
          Browse Files
        </button>
      {/if}
      
      <input 
        type="file" 
        accept=".jpg,.jpeg,.png" 
        style="display: none" 
        bind:this={fileInput} 
        on:change={handleFileChange}
      />
    </div>
    
    <!-- Analysis option -->
    <div class="option-checkbox">
      <input type="checkbox" id="useNeuralrad" bind:checked={useGemini}>
      <label for="useNeuralrad">Use Neuralrad AI for advanced mammogram interpretation</label>
    </div>
    {#if useGemini}
      <div class="option-description">
        Neuralrad AI will provide detailed radiologist-like analysis and check for additional suspicious areas
      </div>
    {/if}
    
    <!-- Error message -->
    {#if error}
      <div class="error-message">
        <p>{error}</p>
      </div>
    {/if}
    
    <!-- Upload button -->
    <div class="analyze-btn-container">
      <button 
        class="analyze-btn" 
        on:click={uploadImage} 
        disabled={!file || uploading}
      >
        {#if uploading}
          <svg class="animate-spin" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="12" y1="2" x2="12" y2="6"></line>
            <line x1="12" y1="18" x2="12" y2="22"></line>
            <line x1="4.93" y1="4.93" x2="7.76" y2="7.76"></line>
            <line x1="16.24" y1="16.24" x2="19.07" y2="19.07"></line>
            <line x1="2" y1="12" x2="6" y2="12"></line>
            <line x1="18" y1="12" x2="22" y2="12"></line>
            <line x1="4.93" y1="19.07" x2="7.76" y2="16.24"></line>
            <line x1="16.24" y1="7.76" x2="19.07" y2="4.93"></line>
          </svg>
          Processing...
        {:else}
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
          </svg>
          Analyze Mammogram
        {/if}
      </button>
    </div>
    
    <!-- Example Images -->
    <div class="examples-row">
      <div class="example-card">
        <img src="/examples/P_00017_LEFT_CC_1.jpg" alt="Example mammogram 1" />
        <div class="example-caption">
          <h3>Sample Mammogram</h3>
          <p>Craniocaudal (CC) view</p>
        </div>
      </div>
      
      <div class="example-card">
        <img src="/examples/P_00037_RIGHT_CC_1.jpg" alt="Example mammogram 2" />
        <div class="example-caption">
          <h3>Sample Mammogram</h3>
          <p>Craniocaudal (CC) view</p>
        </div>
      </div>
      
      <div class="example-card">
        <img src="/examples/example_result.jpg" alt="Example results" />
        <div class="example-caption">
          <h3>AI Detection</h3>
          <p>Sample analysis result</p>
        </div>
      </div>
    </div>
  </div>
{:else if currentTab === 'workflow'}
  <div class="card">
    <h2 class="text-center text-pink">How Neuralrad Mammo AI Works</h2>
    <p class="text-center mb-4">
      This research tool combines deep learning object detection with vision language models for mammogram analysis investigation
    </p>
    
    <div class="workflow-steps">
      {#each workflowSteps as step, index}
        <div class="workflow-step">
          <div class="step-number">{index + 1}</div>
          <div class="step-icon">
            {@html step.icon}
          </div>
          <div class="step-content">
            <h3>{step.title}</h3>
            <p>{step.description}</p>
          </div>
        </div>
      {/each}
    </div>
    
    <div class="text-center mt-4">
      <button class="analyze-btn" on:click={() => switchTab('upload')}>
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
        Upload a Mammogram Now
      </button>
    </div>
  </div>
{/if}
