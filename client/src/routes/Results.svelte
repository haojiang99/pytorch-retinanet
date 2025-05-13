<script>
  import { useNavigate, useLocation } from 'svelte-navigator';
  
  const navigate = useNavigate();
  const location = useLocation();
  
  // Get result data from navigation state
  let result = null;
  
  $: {
    if ($location.state && $location.state.result) {
      result = $location.state.result;
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
              <div style="white-space: pre-line;">
                {result.summary.gemini_analysis}
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
