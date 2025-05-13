<script>
  import { navigate } from 'svelte-navigator';
  
  let file;
  let fileInput;
  let dragging = false;
  let uploading = false;
  let error = null;
  let useGemini = true; // Default to using Gemini analysis
  let currentTab = 'upload'; // 'upload' or 'workflow'
  
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
      description: "Upload a mammogram image (JPEG or PNG format) through our secure interface.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>`
    },
    {
      title: "AI Detection",
      description: "Our first-stage AI identifies and classifies potential masses, highlighting areas of interest.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="2" width="20" height="20" rx="5" ry="5"></rect><path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path><line x1="17.5" y1="6.5" x2="17.51" y2="6.5"></line></svg>`
    },
    {
      title: "Advanced Analysis",
      description: "Neuralrad AI performs a comprehensive analysis to identify suspicious areas that might have been missed.",
      icon: `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>`
    },
    {
      title: "Review Results",
      description: "Examine the comprehensive results with interactive zoom/pan tools and detailed AI analysis.",
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
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 1.5rem;
    margin-top: 3rem;
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
  <h1>Neuralrad Mammo AI</h1>
  <p>Advanced AI-powered mammogram analysis for accurate and reliable breast cancer detection</p>
  
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

{#if currentTab === 'upload'}
  <div class="card">
    <div 
      class="upload-area {dragging ? 'dragging' : ''}" 
      on:drop={handleDrop}
      on:dragover={handleDragOver}
      on:dragleave={handleDragLeave}
    >
      <div class="upload-icon">
        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="17 8 12 3 7 8"></polyline>
          <line x1="12" y1="3" x2="12" y2="15"></line>
        </svg>
      </div>
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
    <div class="text-center">
      <button 
        class="btn btn-primary upload-button" 
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
      Our advanced AI system uses a multi-stage approach to provide accurate and detailed mammogram analysis
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
      <button class="btn btn-primary" on:click={() => switchTab('upload')}>
        Upload a Mammogram Now
      </button>
    </div>
  </div>
{/if}
