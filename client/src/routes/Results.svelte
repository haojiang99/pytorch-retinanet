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
</style>

{#if result}
  <div class="card">
    <h2 class="text-center">Mammography Analysis Results</h2>
    
    <div class="flex" style="gap: 1.5rem; flex-wrap: wrap;">
      <!-- Left column: Image with annotations -->
      <div style="flex: 1; min-width: 400px;">
        <h3>Detected Regions</h3>
        <div style="border: 1px solid #ddd; border-radius: 4px; overflow: hidden;">
          <img 
            src={result.image_data} 
            alt="Mammogram with detections" 
            style="width: 100%; height: auto; display: block;" 
          />
        </div>
      </div>
      
      <!-- Right column: Analysis summary -->
      <div style="flex: 1; min-width: 300px;">
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
              <strong>Total masses detected:</strong> {result.summary.total}
            </p>
            <p>
              <strong>Classification breakdown:</strong>
            </p>
            <ul>
              <li><span class="benign">Benign masses:</span> {result.summary.benign}</li>
              <li><span class="malignant">Malignant masses:</span> {result.summary.malignant}</li>
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
          
          {#if result.summary.gemini_analysis}
            <div class="p-4 mt-4" style="background-color: #f3e5f5; border-radius: 4px; border-left: 4px solid #9c27b0;">
              <h4 style="color: #6a1b9a;">Gemini AI Radiologist Analysis</h4>
              <div class="gemini-content">
                {@html geminiHtml}
              </div>
            </div>
          {/if}
        {/if}
      </div>
    </div>
    
    <div class="text-center mt-4">
      <button class="btn" on:click={goBack}>
        Upload Another Image
      </button>
    </div>
  </div>
{:else}
  <div class="text-center">
    <p>Loading...</p>
  </div>
{/if}
