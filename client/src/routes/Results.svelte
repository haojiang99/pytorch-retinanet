<script>
  import { useNavigate, useLocation } from 'svelte-navigator';
  import { onMount } from 'svelte';
  import { marked } from 'marked';
  import DOMPurify from 'dompurify';
  
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get result data from navigation state
  let result = null;
  let geminiHtml = '';
  
  // Image viewer state
  let scale = 1;
  let translateX = 0;
  let translateY = 0;
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let imageContainer;
  
  $: {
    if ($location.state && $location.state.result) {
      result = $location.state.result;
      
      // Convert Gemini markdown to HTML if available
      if (result.summary && result.summary.gemini_analysis) {
        geminiHtml = DOMPurify.sanitize(marked.parse(result.summary.gemini_analysis));
      }
    } else {
      // If no result data, redirect to home
      navigate('/');
    }
  }
  
  // Image manipulation functions
  function zoomIn() {
    scale = Math.min(scale + 0.25, 5);
  }
  
  function zoomOut() {
    scale = Math.max(scale - 0.25, 0.5);
    // Reset position if zoomed out completely
    if (scale <= 1) {
      translateX = 0;
      translateY = 0;
    }
  }
  
  function resetZoom() {
    scale = 1;
    translateX = 0;
    translateY = 0;
  }
  
  function handleMouseDown(event) {
    if (scale > 1) {
      isDragging = true;
      startX = event.clientX - translateX;
      startY = event.clientY - translateY;
      event.preventDefault();
    }
  }
  
  function handleMouseMove(event) {
    if (isDragging && scale > 1) {
      translateX = event.clientX - startX;
      translateY = event.clientY - startY;
      event.preventDefault();
    }
  }
  
  function handleMouseUp() {
    isDragging = false;
  }
  
  function handleWheel(event) {
    event.preventDefault();
    
    // Get mouse position relative to the image
    const rect = event.currentTarget.getBoundingClientRect();
    const mouseX = event.clientX - rect.left;
    const mouseY = event.clientY - rect.top;
    
    // Determine zoom direction
    const zoomFactor = event.deltaY < 0 ? 0.1 : -0.1;
    const newScale = Math.max(0.5, Math.min(5, scale + zoomFactor));
    
    // Only proceed if scale is changing
    if (newScale !== scale) {
      // Calculate new translate values to zoom toward mouse position
      const scaleChange = newScale / scale;
      
      // Adjust position to zoom toward cursor
      translateX = mouseX - (mouseX - translateX) * scaleChange;
      translateY = mouseY - (mouseY - translateY) * scaleChange;
      
      // Update scale
      scale = newScale;
    }
  }
  
  // Format confidence score
  function formatScore(score) {
    return (score * 100).toFixed(1) + '%';
  }
  
  // Go back to home page
  function goBack() {
    navigate('/');
  }
</script>

<style>
  /* Hero section styles */
  .hero {
    background: linear-gradient(135deg, rgba(233, 30, 99, 0.9) 0%, rgba(156, 39, 176, 0.9) 100%);
    color: white;
    padding: 2.5rem 2rem;
    border-radius: 8px;
    margin-bottom: 2rem;
    text-align: center;
  }
  
  .results-hero {
    padding: 2rem;
  }
  
  .hero h1 {
    font-size: 2.5rem;
    margin-bottom: 0;
    font-weight: 700;
  }
  
  .logo-container {
    margin-bottom: 1rem;
  }
  
  .logo {
    max-width: 160px;
    height: auto;
    border: none;
    box-shadow: none;
  }
  
  /* Style for the Gemini markdown content */
  :global(.gemini-content h2) {
    color: #6a1b9a;
    margin-top: 1.5rem;
    font-size: 1.5rem;
    border-bottom: 2px solid #e1bee7;
    padding-bottom: 0.5rem;
  }
  
  :global(.gemini-content h3) {
    color: #8e24aa;
    margin-top: 1.2rem;
    font-size: 1.2rem;
  }
  
  :global(.gemini-content ul) {
    padding-left: 1.5rem;
  }
  
  :global(.gemini-content li) {
    margin-bottom: 0.5rem;
  }
  
  :global(.gemini-content p) {
    margin-bottom: 1rem;
    line-height: 1.5;
  }
  
  :global(.gemini-content strong) {
    font-weight: 600;
    color: #4a148c;
  }
  
  :global(.gemini-content blockquote) {
    border-left: 3px solid #9c27b0;
    padding-left: 1rem;
    margin-left: 0;
    color: #666;
    font-style: italic;
  }
  
  .benign {
    color: #43a047;
    font-weight: bold;
  }
  
  .malignant {
    color: #d32f2f;
    font-weight: bold;
  }
  
  .benign-calc {
    color: #9e9d24;
    font-weight: bold;
  }
  
  .malignant-calc {
    color: #d84315;
    font-weight: bold;
  }
  
  .image-container {
    position: relative;
    overflow: hidden;
    border: 1px solid #ddd;
    border-radius: 4px;
    width: 100%;
    height: auto;
  }
  
  .image-wrapper {
    cursor: grab;
    transform-origin: 0 0;
    width: 100%;
  }
  
  .image-wrapper.grabbing {
    cursor: grabbing;
  }
  
  .toolbar {
    display: flex;
    gap: 5px;
    margin-bottom: 10px;
  }
  
  .tool-btn {
    background-color: #f3e5f5;
    border: 1px solid #ce93d8;
    border-radius: 4px;
    padding: 5px 10px;
    font-size: 14px;
    cursor: pointer;
    color: #6a1b9a;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  
  .tool-btn:hover {
    background-color: #e1bee7;
  }
  
  .zoom-text {
    display: inline-block;
    padding: 5px 10px;
    font-size: 14px;
    color: #6a1b9a;
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
    min-width: 280px;
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
  
  /* Force the two column layout */
  .two-column-layout {
    display: flex;
    gap: 1.5rem;
    flex-wrap: nowrap !important;
  }
  
  .column {
    flex: 1;
    min-width: 0;
  }
  
  .left-column {
    width: 50%;
  }
  
  .right-column {
    width: 50%;
  }
  
  @media (max-width: 768px) {
    /* On mobile, still try to maintain side-by-side */
    .two-column-layout {
      flex-direction: row;
      overflow-x: auto;
    }
    
    .column {
      min-width: 300px;
      flex-shrink: 0;
    }
  }
</style>

{#if result}
  <div class="hero results-hero">
    <div class="logo-container">
      <img src="/examples/NeuralRadLogo.png" alt="NeuralRad Logo" class="logo" />
    </div>
    <h1>Mammography Research Analysis Results</h1>
  </div>

  <!-- FDA Disclaimer -->
  <div class="disclaimer">
    <h4>⚠️ Research Results Only</h4>
    <p><strong>These results are for research investigation purposes only.</strong> This application is NOT FDA 510(k) cleared and should not be used for clinical diagnosis or treatment decisions.</p>
  </div>

  <div class="card">
    <div class="two-column-layout">
      <!-- Left column: Image with annotations -->
      <div class="column left-column">
        <h3>Detected Regions</h3>
        
        <!-- Image toolbar -->
        <div class="toolbar">
          <button class="tool-btn" on:click={zoomIn}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              <line x1="11" y1="8" x2="11" y2="14"></line>
              <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
          </button>
          <button class="tool-btn" on:click={zoomOut}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
              <line x1="8" y1="11" x2="14" y2="11"></line>
            </svg>
          </button>
          <button class="tool-btn" on:click={resetZoom}>
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M3 12a9 9 0 1 0 18 0 9 9 0 0 0-18 0z"></path>
              <path d="M14 8l-6 8"></path>
              <path d="M10 8l-6 0"></path>
              <path d="M8 12l-4 0"></path>
              <path d="M10 16l-6 0"></path>
            </svg>
          </button>
          <span class="zoom-text">{Math.round(scale * 100)}%</span>
        </div>
        
        <!-- Image container with zoom and pan -->
        <div 
          class="image-container" 
          bind:this={imageContainer}
          on:wheel={handleWheel}
        >
          <div 
            class="image-wrapper {isDragging ? 'grabbing' : ''}" 
            style="transform: scale({scale}) translate({translateX}px, {translateY}px);"
            on:mousedown={handleMouseDown}
            on:mousemove={handleMouseMove}
            on:mouseup={handleMouseUp}
            on:mouseleave={handleMouseUp}
          >
            <img 
              src={result.image_data} 
              alt="Mammogram with detections" 
              style="width: 100%; height: auto; display: block;" 
            />
          </div>
        </div>
        <p class="mt-2" style="font-size: 12px; color: #666; text-align: center;">
          Tip: Use mouse wheel to zoom, click and drag to pan when zoomed
        </p>
      </div>
      
      <!-- Right column: Analysis summary -->
      <div class="column right-column">
        <h3>Analysis Summary</h3>
        
        {#if result.summary.total === 0}
          <div class="p-4" style="background-color: #e8f5e9; border-radius: 4px;">
            <p><strong>No significant findings detected.</strong></p>
            <p>The AI model did not detect any masses in this mammogram.</p>
          </div>
        {:else}
          <div class="p-4 mb-4" style="background-color: #f5f5f5; border-radius: 4px;">
            <h4>Detection Summary</h4>
            <p>
              <strong>Total findings detected:</strong> {result.summary.total}
            </p>
            <p>
              <strong>Classification breakdown:</strong>
            </p>
            <ul>
              <li><span class="benign">Benign masses:</span> {result.summary.mass_benign || 0}</li>
              <li><span class="malignant">Malignant masses:</span> {result.summary.mass_malignant || 0}</li>
              <li><span class="benign-calc">Benign calcifications:</span> {result.summary.calc_benign || 0}</li>
              <li><span class="malignant-calc">Malignant calcifications:</span> {result.summary.calc_malignant || 0}</li>
            </ul>
          </div>
          
          {#if result.summary.highest_confidence}
            <div class="p-4 mb-4" style="background-color: #e3f2fd; border-radius: 4px;">
              <h4>Highest Confidence Detection</h4>
              <p>
                <strong>Classification:</strong> 
                <span class={result.summary.highest_confidence.class}>
                  {result.summary.highest_confidence.class}
                </span>
              </p>
              <p>
                <strong>Confidence:</strong> {formatScore(result.summary.highest_confidence.score)}
              </p>
            </div>
          {/if}
          
          {#if result.summary.largest_mass && result.summary.largest_mass !== result.summary.highest_confidence}
            <div class="p-4 mb-4" style="background-color: #fff3e0; border-radius: 4px;">
              <h4>Largest Mass</h4>
              <p>
                <strong>Classification:</strong> 
                <span class={result.summary.largest_mass.class}>
                  {result.summary.largest_mass.class}
                </span>
              </p>
              <p>
                <strong>Confidence:</strong> {formatScore(result.summary.largest_mass.score)}
              </p>
            </div>
          {/if}
          
          {#if result.summary.findings && result.summary.findings.length > 0}
            <div class="p-4" style="background-color: #f5f5f5; border-radius: 4px;">
              <h4>All Findings</h4>
              {#each result.summary.findings as finding, i}
                <div class="mb-4" style="border-bottom: 1px solid #ddd; padding-bottom: 0.5rem;">
                  <p>
                    <strong>Finding {i+1}:</strong> 
                    <span class={finding.class}>{finding.class}</span>
                  </p>
                  <p>
                    <strong>Confidence:</strong> {formatScore(finding.score)}
                  </p>
                </div>
              {/each}
            </div>
          {/if}
        {/if}
      </div>
    </div>
    
    <!-- Neuralrad AI Analysis Section (Full Width) -->
    {#if result.summary.gemini_analysis}
      <div class="p-4 mt-4" style="background-color: #f3e5f5; border-radius: 4px; border-left: 4px solid #9c27b0;">
        <h3 style="color: #6a1b9a;">Neuralrad AI Analysis</h3>
        <div class="gemini-content">
          {@html geminiHtml}
        </div>
      </div>
    {/if}
    
    <div class="text-center mt-4">
      <button class="analyze-btn" on:click={goBack}>
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path>
          <path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path>
        </svg>
        Upload Another Image
      </button>
    </div>
  </div>
{:else}
  <div class="text-center">
    <p>Loading...</p>
  </div>
{/if}
